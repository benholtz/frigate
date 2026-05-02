import logging
import os
import queue
import subprocess
import threading
import urllib.request
from functools import partial
from multiprocessing import Queue
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import ConfigDict, Field
from typing_extensions import Literal

from frigate.const import MODEL_CACHE_DIR
from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import (
    BaseDetectorConfig,
)
from frigate.object_detection.util import RequestStore, ResponseStore

logger = logging.getLogger(__name__)

# Hailo Model Zoo — precompiled ArcFace MobileFaceNet HEFs. Same S3 bucket
# Frigate already pulls yolov6n.hef from. 99.4% LFW accuracy at INT8 (matches
# float reference; no measurable quantisation loss). MIT-licensed (InsightFace
# upstream).
_ARCFACE_HEF_FILENAME = "arcface_mobilefacenet.hef"
_ARCFACE_HEF_URL_FMT = (
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/"
    "ModelZoo/Compiled/v2.18.0/{arch}/arcface_mobilefacenet.hef"
)


# ----------------- Utility Functions ----------------- #


def preprocess_tensor(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    """
    Resize an image with unchanged aspect ratio using padding.
    Assumes input image shape is (H, W, 3).
    """
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    h, w = image.shape[:2]
    scale = min(model_w / w, model_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    padded_image = np.full((model_h, model_w, 3), 114, dtype=image.dtype)
    x_offset = (model_w - new_w) // 2
    y_offset = (model_h - new_h) // 2
    padded_image[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
        resized_image
    )
    return padded_image


# ----------------- Global Constants ----------------- #
DETECTOR_KEY = "hailo8l"
ARCH = None
H8_DEFAULT_MODEL = "yolov6n.hef"
H8L_DEFAULT_MODEL = "yolov6n.hef"
H8_DEFAULT_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov6n.hef"
H8L_DEFAULT_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov6n.hef"


def detect_hailo_arch():
    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"], capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error(f"Inference error: {result.stderr}")
            return None
        for line in result.stdout.split("\n"):
            if "Device Architecture" in line:
                if "HAILO8L" in line:
                    return "hailo8l"
                elif "HAILO8" in line:
                    return "hailo8"
        logger.error("Inference error: Could not determine Hailo architecture.")
        return None
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return None


# ----------------- HailoAsyncInference Class ----------------- #
class HailoAsyncInference:
    def __init__(
        self,
        hef_path: str,
        input_store: RequestStore,
        output_store: ResponseStore,
        batch_size: int = 1,
        input_type: Optional[str] = None,
        output_type: Optional[Dict[str, str]] = None,
        send_original_frame: bool = False,
    ) -> None:
        # when importing hailo it activates the driver
        # which leaves processes running even though it may not be used.
        try:
            from hailo_platform import (
                HEF,
                FormatType,
                HailoSchedulingAlgorithm,
                VDevice,
            )
        except ModuleNotFoundError:
            pass

        self.input_store = input_store
        self.output_store = output_store

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)

        if input_type is not None:
            self.infer_model.input().set_format_type(getattr(FormatType, input_type))

        if output_type is not None:
            for output_name, output_type in output_type.items():
                self.infer_model.output(output_name).set_format_type(
                    getattr(FormatType, output_type)
                )

        self.output_type = output_type
        self.send_original_frame = send_original_frame

    def callback(
        self,
        completion_info,
        bindings_list: List,
        input_batch: List,
        request_ids: List[int],
    ):
        if completion_info.exception:
            logger.error(f"Inference error: {completion_info.exception}")
        else:
            for i, bindings in enumerate(bindings_list):
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                        for name in bindings._output_names
                    }
                self.output_store.put(request_ids[i], (input_batch[i], result))

    def _create_bindings(self, configured_infer_model) -> object:
        if self.output_type is None:
            output_buffers = {
                output_info.name: np.empty(
                    self.infer_model.output(output_info.name).shape,
                    dtype=getattr(
                        np, str(output_info.format.type).split(".")[1].lower()
                    ),
                )
                for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape,
                    dtype=getattr(np, self.output_type[name].lower()),
                )
                for name in self.output_type
            }
        return configured_infer_model.create_bindings(output_buffers=output_buffers)

    def get_input_shape(self) -> Tuple[int, ...]:
        return self.hef.get_input_vstream_infos()[0].shape

    def run(self) -> None:
        job = None
        with self.infer_model.configure() as configured_infer_model:
            while True:
                batch_data = self.input_store.get()

                if batch_data is None:
                    break

                request_id, frame_data = batch_data
                preprocessed_batch = [frame_data]
                request_ids = [request_id]
                input_batch = preprocessed_batch  # non-send_original_frame mode

                bindings_list = []
                for frame in preprocessed_batch:
                    bindings = self._create_bindings(configured_infer_model)
                    bindings.input().set_buffer(np.array(frame))
                    bindings_list.append(bindings)
                configured_infer_model.wait_for_async_ready(timeout_ms=10000)
                job = configured_infer_model.run_async(
                    bindings_list,
                    partial(
                        self.callback,
                        input_batch=input_batch,
                        request_ids=request_ids,
                        bindings_list=bindings_list,
                    ),
                )

            if job is not None:
                job.wait(100)


# ----------------- ArcFace face inference engine ----------------- #
class _HailoFaceInferenceEngine:
    """Synchronous ArcFace runner that uses an externally-owned VDevice.

    Used by HailoDetector to satisfy face embedding requests forwarded
    from the embeddings worker process, without claiming a second VDevice.
    The YOLO inference engine and this engine share the same VDevice; the
    HailoRT scheduler (HailoSchedulingAlgorithm.ROUND_ROBIN, set on the
    VDevice when HailoAsyncInference creates it) multiplexes the two
    models on the chip.
    """

    def __init__(self, vdevice, hef_path: str):
        try:
            from hailo_platform import HEF, FormatType
        except ModuleNotFoundError:
            raise

        self._vdevice = vdevice
        self._hef_path = hef_path
        # Hold a HEF reference alongside the InferModel — output enumeration
        # goes through HEF.get_output_vstream_infos(), not a private attr
        # on InferModel.
        self._hef = HEF(hef_path)
        self._infer_model = vdevice.create_infer_model(hef_path)
        self._infer_model.set_batch_size(1)
        self._infer_model.input().set_format_type(FormatType.UINT8)
        self._output_names = [
            info.name for info in self._hef.get_output_vstream_infos()
        ]

        self._configured_cm = None
        self._configured = None

    def _ensure_configured(self) -> None:
        if self._configured is None:
            self._configured_cm = self._infer_model.configure()
            self._configured = self._configured_cm.__enter__()

    def infer(self, nhwc_uint8: np.ndarray) -> np.ndarray:
        self._ensure_configured()

        output_buffers = {
            name: np.empty(
                self._infer_model.output(name).shape,
                dtype=np.float32,
            )
            for name in self._output_names
        }

        bindings = self._configured.create_bindings(
            output_buffers=output_buffers
        )
        bindings.input().set_buffer(np.ascontiguousarray(nhwc_uint8[0]))

        self._configured.wait_for_async_ready(timeout_ms=10000)

        done = threading.Event()
        error_holder: Dict[str, Optional[Exception]] = {"err": None}

        def _cb(completion_info):
            if completion_info.exception:
                error_holder["err"] = completion_info.exception
            done.set()

        job = self._configured.run_async([bindings], _cb)
        job.wait(10000)
        done.wait(timeout=10.0)

        if error_holder["err"] is not None:
            raise RuntimeError(
                f"ArcFace inference failed: {error_holder['err']}"
            )

        # Reshape the (only) output buffer to (1, 512) to match the ONNX
        # ArcFace model's output contract.
        embedding = next(iter(output_buffers.values()))
        return np.asarray(embedding, dtype=np.float32).reshape((1, -1))

    def close(self) -> None:
        if self._configured_cm is not None:
            try:
                self._configured_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._configured = None
            self._configured_cm = None


# ----------------- HailoDetector Class ----------------- #
class HailoDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: "HailoDetectorConfig"):
        global ARCH
        ARCH = detect_hailo_arch()
        self.cache_dir = MODEL_CACHE_DIR
        self.device_type = detector_config.device
        self.model_height = (
            detector_config.model.height
            if hasattr(detector_config.model, "height")
            else None
        )
        self.model_width = (
            detector_config.model.width
            if hasattr(detector_config.model, "width")
            else None
        )
        self.model_type = (
            detector_config.model.model_type
            if hasattr(detector_config.model, "model_type")
            else None
        )
        self.tensor_format = (
            detector_config.model.input_tensor
            if hasattr(detector_config.model, "input_tensor")
            else None
        )
        self.pixel_format = (
            detector_config.model.input_pixel_format
            if hasattr(detector_config.model, "input_pixel_format")
            else None
        )
        self.input_dtype = (
            detector_config.model.input_dtype
            if hasattr(detector_config.model, "input_dtype")
            else None
        )
        self.output_type = "FLOAT32"
        self.set_path_and_url(detector_config.model.path)
        self.working_model_path = self.check_and_prepare()

        self.batch_size = 1
        self.input_store = RequestStore()
        self.response_store = ResponseStore()

        try:
            logger.debug(f"[INIT] Loading HEF model from {self.working_model_path}")
            self.inference_engine = HailoAsyncInference(
                self.working_model_path,
                self.input_store,
                self.response_store,
                self.batch_size,
            )
            self.input_shape = self.inference_engine.get_input_shape()
            logger.debug(f"[INIT] Model input shape: {self.input_shape}")
            self.inference_thread = threading.Thread(
                target=self.inference_engine.run, daemon=True
            )
            self.inference_thread.start()
        except Exception as e:
            logger.error(f"[INIT] Failed to initialize HailoAsyncInference: {e}")
            raise

        # Face recognition proxy state (populated lazily by
        # set_face_inference_queues when the embeddings worker is also
        # configured for device=hailo).
        self._face_engine: Optional[_HailoFaceInferenceEngine] = None
        self._face_request_queue: Optional[Queue] = None
        self._face_response_queue: Optional[Queue] = None
        self._face_worker: Optional[threading.Thread] = None
        self._face_stop = threading.Event()

    def set_face_inference_queues(
        self, request_queue: Queue, response_queue: Queue
    ) -> None:
        """Co-locate ArcFace face inference on this detector's VDevice and
        consume face crops from the embeddings worker via the queue pair.

        Called from BaseLocalDetector.__init__ via duck-typing (mirrors
        the existing set_stop_event pattern at base.py line 67-68) when
        face_recognition.device is set to "hailo". Hailo-8 / 8L allow
        only one VDevice handle per chip per process, so the embeddings
        worker process cannot run ArcFace itself; this method loads the
        ArcFace HEF on the YOLO VDevice via HailoRT's ROUND_ROBIN
        scheduler and serves face inferences in a daemon worker thread.
        """
        if self._face_worker is not None:
            return  # already started

        hef_dir = os.path.join(MODEL_CACHE_DIR, "facedet")
        hef_path = os.path.join(hef_dir, _ARCFACE_HEF_FILENAME)
        if not os.path.exists(hef_path):
            os.makedirs(hef_dir, exist_ok=True)
            arch = ARCH if ARCH in ("hailo8", "hailo8l") else "hailo8l"
            url = _ARCFACE_HEF_URL_FMT.format(arch=arch)
            logger.info(f"[face-proxy] Downloading ArcFace HEF from {url}")
            urllib.request.urlretrieve(url, hef_path)

        try:
            self._face_engine = _HailoFaceInferenceEngine(
                self.inference_engine.target, hef_path
            )
        except Exception as e:
            logger.error(
                f"[face-proxy] Failed to load ArcFace HEF on VDevice: {e}. "
                "Detector will continue serving YOLO; face_recognition will "
                "not run on the NPU."
            )
            self._face_engine = None
            return

        self._face_request_queue = request_queue
        self._face_response_queue = response_queue
        self._face_worker = threading.Thread(
            target=self._face_inference_worker,
            name="hailo-face-worker",
            daemon=True,
        )
        self._face_worker.start()
        logger.info(
            "[face-proxy] ArcFace face recognition proxy ready "
            "(co-located with YOLO on the Hailo VDevice)"
        )

    def _face_inference_worker(self) -> None:
        """Daemon worker: pull face crops from the request queue, run
        ArcFace on the shared VDevice, push the embedding to the
        response queue. Crops are uint8 NHWC of shape (112, 112, 3).
        Responses are tuples of (status, payload):
          (0, np.ndarray)  on success — embedding shape (1, 512)
          (1, str)         on failure — error message
        """
        while not self._face_stop.is_set():
            try:
                tensor = self._face_request_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if tensor is None:
                # Shutdown sentinel from the embeddings worker side.
                break

            try:
                # _HailoFaceInferenceEngine.infer expects NHWC (1, H, W, 3).
                if tensor.ndim == 3:
                    tensor = tensor[np.newaxis, ...]
                embedding = self._face_engine.infer(tensor)
                self._face_response_queue.put((0, embedding))
            except Exception as e:
                logger.error(f"[face-proxy] inference failed: {e}")
                try:
                    self._face_response_queue.put((1, str(e)))
                except Exception:
                    pass

    def set_path_and_url(self, path: str = None):
        if not path:
            self.model_path = None
            self.url = None
            return
        if self.is_url(path):
            self.url = path
            self.model_path = None
        else:
            self.model_path = path
            self.url = None

    def is_url(self, url: str) -> bool:
        return (
            url.startswith("http://")
            or url.startswith("https://")
            or url.startswith("www.")
        )

    @staticmethod
    def extract_model_name(path: str = None, url: str = None) -> str:
        if path and path.endswith(".hef"):
            return os.path.basename(path)
        elif url and url.endswith(".hef"):
            return os.path.basename(url)
        else:
            if ARCH == "hailo8":
                return H8_DEFAULT_MODEL
            else:
                return H8L_DEFAULT_MODEL

    @staticmethod
    def download_model(url: str, destination: str):
        if not url.endswith(".hef"):
            raise ValueError("Invalid model URL. Only .hef files are supported.")
        try:
            urllib.request.urlretrieve(url, destination)
            logger.debug(f"Downloaded model to {destination}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model from {url}: {str(e)}")

    def check_and_prepare(self) -> str:
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        model_name = self.extract_model_name(self.model_path, self.url)
        cached_model_path = os.path.join(self.cache_dir, model_name)
        if not self.model_path and not self.url:
            if os.path.exists(cached_model_path):
                logger.debug(f"Model found in cache: {cached_model_path}")
                return cached_model_path
            else:
                logger.debug(f"Downloading default model: {model_name}")
                if ARCH == "hailo8":
                    self.download_model(H8_DEFAULT_URL, cached_model_path)
                else:
                    self.download_model(H8L_DEFAULT_URL, cached_model_path)
        elif self.url:
            logger.debug(f"Downloading model from URL: {self.url}")
            self.download_model(self.url, cached_model_path)
        elif self.model_path:
            if os.path.exists(self.model_path):
                logger.debug(f"Using existing model at: {self.model_path}")
                return self.model_path
            else:
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        return cached_model_path

    def detect_raw(self, tensor_input):
        tensor_input = self.preprocess(tensor_input)

        if isinstance(tensor_input, np.ndarray) and len(tensor_input.shape) == 3:
            tensor_input = np.expand_dims(tensor_input, axis=0)

        request_id = self.input_store.put(tensor_input)

        try:
            _, infer_results = self.response_store.get(request_id, timeout=1.0)
        except TimeoutError:
            logger.error(
                f"Timeout waiting for inference results for request {request_id}"
            )

            if not self.inference_thread.is_alive():
                raise RuntimeError(
                    "HailoRT inference thread has stopped, restart required."
                )

            return np.zeros((20, 6), dtype=np.float32)

        if isinstance(infer_results, list) and len(infer_results) == 1:
            infer_results = infer_results[0]

        threshold = 0.4
        all_detections = []
        for class_id, detection_set in enumerate(infer_results):
            if not isinstance(detection_set, np.ndarray) or detection_set.size == 0:
                continue
            for det in detection_set:
                if det.shape[0] < 5:
                    continue
                score = float(det[4])
                if score < threshold:
                    continue
                all_detections.append([class_id, score, det[0], det[1], det[2], det[3]])

        if len(all_detections) == 0:
            detections_array = np.zeros((20, 6), dtype=np.float32)
        else:
            detections_array = np.array(all_detections, dtype=np.float32)
            if detections_array.shape[0] > 20:
                detections_array = detections_array[:20, :]
            elif detections_array.shape[0] < 20:
                pad = np.zeros((20 - detections_array.shape[0], 6), dtype=np.float32)
                detections_array = np.vstack((detections_array, pad))

        return detections_array

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            processed = preprocess_tensor(
                image, self.input_shape[1], self.input_shape[0]
            )
            return np.expand_dims(processed, axis=0)
        else:
            raise ValueError("Unsupported image format for preprocessing")

    def close(self):
        """Properly shuts down the inference engine and releases the VDevice."""
        logger.debug("[CLOSE] Closing HailoDetector")
        try:
            if hasattr(self, "inference_engine"):
                if hasattr(self.inference_engine, "target"):
                    self.inference_engine.target.release()
                logger.debug("Hailo VDevice released successfully")
        except Exception as e:
            logger.error(f"Failed to close Hailo device: {e}")
            raise

    def __del__(self):
        """Destructor to ensure cleanup when the object is deleted."""
        self.close()


# ----------------- HailoDetectorConfig Class ----------------- #
class HailoDetectorConfig(BaseDetectorConfig):
    """Hailo-8/Hailo-8L detector using HEF models and the HailoRT SDK for inference on Hailo hardware."""

    model_config = ConfigDict(
        title="Hailo-8/Hailo-8L",
    )

    type: Literal[DETECTOR_KEY]
    device: str = Field(
        default="PCIe",
        title="Device Type",
        description="The device to use for Hailo inference (e.g. 'PCIe', 'M.2').",
    )

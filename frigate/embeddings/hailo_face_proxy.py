"""IPC proxy that lets the embeddings worker run ArcFace on the Hailo NPU.

Background
----------
The Hailo-8 / Hailo-8L exposes a single physical chip and HailoRT only allows
one VDevice handle per chip per process. Frigate's detector process already
owns that VDevice for the YOLO object detector. When the embeddings worker
tries to claim its own VDevice for ArcFace face embeddings, HailoRT returns
HAILO_OUT_OF_PHYSICAL_DEVICES (74).

This module solves the conflict by making the detector process the only
owner of the VDevice and exposing a tiny TCP service that the embeddings
worker calls into to run face inference. HailoRT's ROUND_ROBIN scheduler
multiplexes the two models (YOLO + ArcFace) on the shared VDevice.

The proxy is loopback-only (127.0.0.1) - it is not a network surface.

Wire format
-----------
Request (client -> server):
    uint32 BE: request_id
    uint32 BE: payload length in bytes
    bytes:     uint8 NHWC face crop, 1 x 112 x 112 x 3 = 37632 bytes
Response (server -> client):
    uint32 BE: request_id (echo)
    uint32 BE: status (0 = ok, non-zero = error code)
    uint32 BE: payload length in bytes
    bytes:     float32 embedding, 1 x 512 = 2048 bytes (when status == 0)
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Loopback-only port. Both processes are inside the same container so 127.0.0.1
# is the only interface either can reach. Hardcoded to keep configuration zero
# for the user — the only knob is whether face_recognition.device == "hailo".
HAILO_FACE_PROXY_HOST = "127.0.0.1"
HAILO_FACE_PROXY_PORT = 5005

# ArcFace MobileFaceNet input/output shape constants (Hailo HEF expects NHWC
# uint8 [0,255]; the model produces a 512-dim float32 embedding).
FACE_CROP_BYTES = 1 * 112 * 112 * 3
EMBEDDING_FLOATS = 512
EMBEDDING_BYTES = EMBEDDING_FLOATS * 4

_HEADER_FMT = ">II"  # request_id, payload_length
_RESPONSE_FMT = ">III"  # request_id, status, payload_length
_HEADER_LEN = struct.calcsize(_HEADER_FMT)
_RESPONSE_LEN = struct.calcsize(_RESPONSE_FMT)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from a socket, looping until we have them."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Hailo face proxy peer closed the connection")
        buf.extend(chunk)
    return bytes(buf)


class HailoFaceProxyServer:
    """Runs inside the detector process. Owns the ArcFace inference engine on
    the same VDevice the YOLO detector uses. Listens on a loopback TCP socket
    for inference requests from the embeddings worker.

    The caller is responsible for providing an already-initialised HailoRT
    inference engine for ArcFace (a HailoAsyncInference instance is the most
    common shape, but anything with `run()` semantics works).
    """

    def __init__(self, infer_callable, host: str = HAILO_FACE_PROXY_HOST,
                 port: int = HAILO_FACE_PROXY_PORT):
        """
        infer_callable: a callable taking a uint8 NHWC ndarray of shape
                        (1, 112, 112, 3) and returning a float32 ndarray of
                        shape (1, 512). Must be thread-safe (or sufficiently
                        synchronised internally).
        """
        self._infer = infer_callable
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._host, self._port))
        self._sock.listen(8)
        self._sock.settimeout(1.0)
        self._thread = threading.Thread(
            target=self._accept_loop,
            name="hailo-face-proxy-server",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Hailo face proxy listening on {self._host}:{self._port} "
            f"(serving ArcFace embeddings to the embeddings worker)"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._sock.accept()  # type: ignore[union-attr]
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(
                target=self._handle_client,
                args=(client,),
                name="hailo-face-proxy-client",
                daemon=True,
            )
            t.start()

    def _handle_client(self, client: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                header = _recv_exact(client, _HEADER_LEN)
                request_id, payload_len = struct.unpack(_HEADER_FMT, header)

                if payload_len != FACE_CROP_BYTES:
                    self._send_error(
                        client, request_id, status=2,
                        msg=f"unexpected payload length {payload_len}",
                    )
                    continue

                payload = _recv_exact(client, payload_len)
                tensor = np.frombuffer(payload, dtype=np.uint8).reshape(
                    (1, 112, 112, 3)
                )

                try:
                    embedding = self._infer(tensor)
                except Exception as e:
                    logger.error(f"Hailo face inference failed: {e}")
                    self._send_error(client, request_id, status=1, msg=str(e))
                    continue

                emb_bytes = np.asarray(
                    embedding, dtype=np.float32
                ).reshape(-1).tobytes()

                if len(emb_bytes) != EMBEDDING_BYTES:
                    logger.error(
                        f"Embedding has unexpected size {len(emb_bytes)}"
                    )
                    self._send_error(client, request_id, status=3,
                                     msg="embedding size mismatch")
                    continue

                response = struct.pack(
                    _RESPONSE_FMT, request_id, 0, len(emb_bytes)
                )
                client.sendall(response + emb_bytes)
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                client.close()
            except Exception:
                pass

    @staticmethod
    def _send_error(client: socket.socket, request_id: int, status: int,
                    msg: str) -> None:
        try:
            err = msg.encode("utf-8")[:1024]
            response = struct.pack(_RESPONSE_FMT, request_id, status, len(err))
            client.sendall(response + err)
        except Exception:
            pass


class HailoFaceProxyClient:
    """Runs inside the embeddings worker. Connects to HailoFaceProxyServer
    over loopback TCP and forwards face crops for inference. Synchronous,
    one-call-at-a-time per instance — suitable for the embeddings_manager
    process which serialises face_recognition work anyway.
    """

    def __init__(self, host: str = HAILO_FACE_PROXY_HOST,
                 port: int = HAILO_FACE_PROXY_PORT,
                 connect_timeout: float = 30.0):
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._next_request_id = 1
        self._connect_timeout = connect_timeout
        self._connect()

    def _connect(self) -> None:
        deadline = self._connect_timeout
        start = None
        last_error: Optional[BaseException] = None
        import time

        start = time.monotonic()
        while time.monotonic() - start < deadline:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(15.0)
                sock.connect((self._host, self._port))
                self._sock = sock
                return
            except (ConnectionRefusedError, OSError) as e:
                last_error = e
                time.sleep(0.5)
        raise RuntimeError(
            f"Could not connect to Hailo face proxy at "
            f"{self._host}:{self._port} within {self._connect_timeout}s "
            f"(last error: {last_error})"
        )

    def infer(self, nhwc_uint8: np.ndarray) -> np.ndarray:
        if nhwc_uint8.dtype != np.uint8:
            raise ValueError("Input must be uint8 NHWC")
        if nhwc_uint8.shape != (1, 112, 112, 3):
            raise ValueError(
                f"Input shape must be (1,112,112,3), got {nhwc_uint8.shape}"
            )

        payload = nhwc_uint8.tobytes()
        if len(payload) != FACE_CROP_BYTES:
            raise ValueError(
                f"Payload length mismatch: got {len(payload)}, "
                f"expected {FACE_CROP_BYTES}"
            )

        with self._lock:
            request_id = self._next_request_id
            self._next_request_id = (self._next_request_id + 1) & 0xFFFFFFFF

            header = struct.pack(_HEADER_FMT, request_id, len(payload))
            assert self._sock is not None
            self._sock.sendall(header + payload)

            response = _recv_exact(self._sock, _RESPONSE_LEN)
            resp_id, status, resp_len = struct.unpack(_RESPONSE_FMT, response)

            if resp_id != request_id:
                raise RuntimeError(
                    f"Hailo face proxy response id mismatch: "
                    f"sent {request_id}, got {resp_id}"
                )

            body = _recv_exact(self._sock, resp_len) if resp_len else b""

            if status != 0:
                raise RuntimeError(
                    f"Hailo face proxy returned status {status}: "
                    f"{body.decode('utf-8', 'replace')}"
                )

            if len(body) != EMBEDDING_BYTES:
                raise RuntimeError(
                    f"Hailo face proxy returned unexpected embedding size "
                    f"{len(body)}, expected {EMBEDDING_BYTES}"
                )

            return np.frombuffer(body, dtype=np.float32).reshape((1, 512))

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

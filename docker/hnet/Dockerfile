# Overlay build: upstream's own rpi image for the v0.18.2 commit (cfa08c5),
# plus the two Python files carrying the no_frames_threshold patch.
#
# v0.18.2 can no longer be rebuilt from source: its nginx stage downloads
# kaltura/nginx-vod-module and kaltura/nginx-secure-token-module, and both
# repositories have been deleted. Overlaying keeps every upstream binary
# exactly as released. The workflow checks that the two files being replaced
# are byte-identical to v0.18.2 in the base image before building.
FROM ghcr.io/blakeblackshear/frigate:cfa08c5-rpi@sha256:7f395bb1db0ff766b387f8c70f85f4d2b5da63ab588693fcda8cc19f5e916cea
ARG HNET_SHA
COPY frigate/config/camera/ffmpeg.py /opt/frigate/frigate/config/camera/ffmpeg.py
COPY frigate/video/ffmpeg.py /opt/frigate/frigate/video/ffmpeg.py
RUN python3 -m py_compile /opt/frigate/frigate/config/camera/ffmpeg.py /opt/frigate/frigate/video/ffmpeg.py \
    && grep -q '^VERSION = "0.18.2-cfa08c5"$' /opt/frigate/frigate/version.py \
    && echo "VERSION = \"0.18.2-${HNET_SHA}\"" > /opt/frigate/frigate/version.py

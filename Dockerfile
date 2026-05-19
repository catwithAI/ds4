# DS4 CUDA build for ARM64 (aarch64 / sbsa) — targets DGX Spark / GB10 and other
# NVIDIA ARM platforms. Requires the host to have NVIDIA drivers and the NVIDIA
# Container Toolkit; run with `--gpus all`.
#
# Build (on an aarch64 host, or with buildx --platform=linux/arm64):
#   docker build -t ds4:cuda-arm64 .
#
# Run (mount your GGUF directory; do NOT bake weights into the image):
#   docker run --rm --gpus all \
#     -p 30001:30001 \
#     -v /path/to/gguf:/models \
#     ds4:cuda-arm64 \
#     ./ds4-server --model /models/ds4flash.gguf --host 0.0.0.0 --port 30001
#
# Override the CUDA arch at build time if you are not on GB10:
#   docker build --build-arg CUDA_ARCH=sm_90  -t ds4:cuda-arm64 .   # Grace Hopper
#   docker build --build-arg CUDA_ARCH=sm_87  -t ds4:cuda-arm64 .   # Orin / AGX
#   docker build --build-arg CUDA_TARGET=cuda-generic -t ds4:cuda-arm64 .   # build on target

# NOTE: GB10 / Blackwell (sm_120) needs CUDA toolkit >= 12.8. CUDA 12.6 nvcc
# rejects -arch=sm_120 with "Value 'sm_120' is not defined for option 'gpu-architecture'".
# 12.8.1 is the most stable Blackwell-capable toolkit at time of writing and
# requires NVIDIA driver >= 570 on the host.
ARG CUDA_IMAGE_TAG=12.8.1-devel-ubuntu22.04
ARG CUDA_RUNTIME_TAG=12.8.1-runtime-ubuntu22.04

# ---- build stage -----------------------------------------------------------
FROM --platform=$BUILDPLATFORM nvidia/cuda:${CUDA_IMAGE_TAG} AS build

# Default to explicit sm_120 (GB10 Blackwell). The Makefile's `cuda-spark`
# target deliberately omits -arch, which is the fastest path *on* GB10 when
# the local nvcc default already matches, but inside this image nvcc defaults
# to a very old arch (sm_52) where __dp4a is not exposed and the build fails.
# We therefore route through `make cuda CUDA_ARCH=sm_120` by default.
ARG CUDA_TARGET=cuda
ARG CUDA_ARCH=sm_120

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

# -march=native is x86-only; on ARM the Makefile already picks -mcpu=native via
# UNAME_S=Linux + aarch64. Override to a portable baseline so the image is not
# pinned to the build host's exact CPU revision.
ENV NATIVE_CPU_FLAG=-mcpu=neoverse-v2

RUN if [ "$CUDA_TARGET" = "cuda" ] && [ -z "$CUDA_ARCH" ]; then \
        echo "CUDA_TARGET=cuda requires --build-arg CUDA_ARCH=sm_XX" >&2; exit 2; \
    fi && \
    if [ -n "$CUDA_ARCH" ]; then \
        make $CUDA_TARGET CUDA_ARCH=$CUDA_ARCH; \
    else \
        make $CUDA_TARGET; \
    fi

# ---- runtime stage ---------------------------------------------------------
FROM nvidia/cuda:${CUDA_RUNTIME_TAG} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/ds4         /app/ds4
COPY --from=build /src/ds4-server  /app/ds4-server
COPY --from=build /src/ds4-bench   /app/ds4-bench
COPY --from=build /src/ds4-eval    /app/ds4-eval
COPY --from=build /src/download_model.sh /app/download_model.sh

ENV PATH=/app:$PATH

EXPOSE 30001
VOLUME ["/models"]

CMD ["./ds4-server", "--host", "0.0.0.0", "--port", "30001", "--model", "/models/ds4flash.gguf"]

#!/usr/bin/env bash
# Build the DS4 CUDA ARM64 image.
#
# Usage:
#   ./build.sh                          # default: sm_120 (DGX Spark / GB10 Blackwell)
#   ./build.sh spark                    # cuda-spark (no -arch; only works when host nvcc matches)
#   ./build.sh generic                  # nvcc -arch=native (build on target)
#   ./build.sh sm_90                    # explicit CUDA arch (Grace Hopper)
#   ./build.sh sm_87                    # Orin / AGX
#
# Env overrides:
#   IMAGE_NAME       image tag (default: ds4:cuda-arm64)
#   CUDA_IMAGE_TAG   devel base image tag  (default: 12.8.1-devel-ubuntu22.04)
#   CUDA_RUNTIME_TAG runtime base tag      (default: 12.8.1-runtime-ubuntu22.04)
#   CPU_FLAG         -mcpu value           (default: neoverse-v2 for GB10)
#   PROGRESS         buildx --progress     (default: auto)
#   SAVE_TAR         set to 0 to skip the docker-save step (default: 1)
#   TAR_PATH         output tar path        (default: ./<image-without-colons>.tar
#                    next to this script, e.g. ./ds4-cuda-arm64.tar)

set -euo pipefail

# Always run from the directory containing this script so the build context
# (and the output tar) are anchored to the repo root regardless of CWD.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="${IMAGE_NAME:-ds4:cuda-arm64}"
CUDA_IMAGE_TAG="${CUDA_IMAGE_TAG:-12.8.1-devel-ubuntu22.04}"
CUDA_RUNTIME_TAG="${CUDA_RUNTIME_TAG:-12.8.1-runtime-ubuntu22.04}"
CPU_FLAG="${CPU_FLAG:-}"
PROGRESS="${PROGRESS:-auto}"
SAVE_TAR="${SAVE_TAR:-1}"
# Default tar path: ./<image-name-with-colon-replaced>.tar in the repo root.
DEFAULT_TAR_NAME="$(echo "$IMAGE_NAME" | tr ':/' '--').tar"
TAR_PATH="${TAR_PATH:-$SCRIPT_DIR/$DEFAULT_TAR_NAME}"

target="${1:-sm_120}"

case "$target" in
    spark|cuda-spark)
        # Note: cuda-spark omits -arch. nvidia/cuda:*-devel images default
        # nvcc to sm_52, which does not expose __dp4a; the build will fail.
        # Use this target only when building outside Docker on the GB10 host
        # itself, where the local nvcc default already matches GB10.
        CUDA_TARGET="cuda-spark"
        CUDA_ARCH=""
        ;;
    generic|cuda-generic)
        CUDA_TARGET="cuda-generic"
        CUDA_ARCH=""
        ;;
    sm_*)
        CUDA_TARGET="cuda"
        CUDA_ARCH="$target"
        ;;
    -h|--help|help)
        sed -n '2,20p' "$0"
        exit 0
        ;;
    *)
        echo "Unknown target: $target" >&2
        echo "Run './build.sh --help' for usage." >&2
        exit 2
        ;;
esac

build_args=(
    --build-arg "CUDA_IMAGE_TAG=$CUDA_IMAGE_TAG"
    --build-arg "CUDA_RUNTIME_TAG=$CUDA_RUNTIME_TAG"
    --build-arg "CUDA_TARGET=$CUDA_TARGET"
)
if [ -n "$CUDA_ARCH" ]; then
    build_args+=( --build-arg "CUDA_ARCH=$CUDA_ARCH" )
fi
if [ -n "$CPU_FLAG" ]; then
    build_args+=( --build-arg "NATIVE_CPU_FLAG=-mcpu=$CPU_FLAG" )
fi

echo "Building $IMAGE_NAME"
echo "  base devel:    $CUDA_IMAGE_TAG"
echo "  base runtime:  $CUDA_RUNTIME_TAG"
echo "  make target:   $CUDA_TARGET${CUDA_ARCH:+ (CUDA_ARCH=$CUDA_ARCH)}"
echo "  CPU flag:      ${CPU_FLAG:-(image default: neoverse-v2)}"
echo

# Prefer buildx when available so the build runs in linux/arm64 even on
# non-ARM hosts (e.g. emulated via QEMU). Falls back to plain `docker build`.
if docker buildx version >/dev/null 2>&1; then
    docker buildx build \
        --platform linux/arm64 \
        --load \
        --progress "$PROGRESS" \
        "${build_args[@]}" \
        -t "$IMAGE_NAME" \
        -f Dockerfile \
        .
else
    echo "warning: docker buildx not found, falling back to 'docker build' (host must be arm64)" >&2
    docker build "${build_args[@]}" -t "$IMAGE_NAME" -f Dockerfile .
fi

echo
echo "Done. Image: $IMAGE_NAME"

if [ "$SAVE_TAR" != "0" ]; then
    echo
    echo "Saving image to $TAR_PATH"
    # docker save streams the layers; show a rough size after it finishes.
    docker save "$IMAGE_NAME" -o "$TAR_PATH"
    if command -v du >/dev/null 2>&1; then
        echo "  size: $(du -h "$TAR_PATH" | awk '{print $1}')"
    fi
    echo "Load on another host with:"
    echo "  docker load -i $(basename "$TAR_PATH")"
else
    echo "(SAVE_TAR=0: skipping docker save)"
fi

echo
echo "Run with:"
echo "  docker compose up -d        # using docker-compose.yaml"
echo "  # or"
echo "  docker run --rm --gpus all -p 30001:30001 \\"
echo "    -v \$PWD/gguf:/models -v \$PWD/kv-cache:/kv \\"
echo "    $IMAGE_NAME"

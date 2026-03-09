#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="openmmlab"
PYTHON_VERSION="3.11"
KEEP_EXISTING_ENV=0
SKIP_DEMO=0

usage() {
    cat <<'EOF'
Usage: bash tools/install_openmmlab_cuda121.sh [options]

Options:
  --env-name NAME          Conda environment name. Default: openmmlab
  --python-version VER     Python version for the conda environment. Default: 3.11
  --keep-existing-env      Reuse the existing conda environment if it already exists
  --skip-demo              Skip the top-down demo smoke test
  -h, --help               Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="${2:?missing value for --env-name}"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="${2:?missing value for --python-version}"
            shift 2
            ;;
        --keep-existing-env)
            KEEP_EXISTING_ENV=1
            shift
            ;;
        --skip-demo)
            SKIP_DEMO=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SMOKE_TEST_IMAGE="${REPO_ROOT}/tests/data/coco/000000000785.jpg"
SMOKE_TEST_OUTPUT="${REPO_ROOT}/vis_results"

TORCH_VERSION="2.1.0"
TORCHVISION_VERSION="0.16.0"
TORCHAUDIO_VERSION="2.1.0"
MMCV_VERSION="2.1.0"
MMENGINE_VERSION="0.10.5"
MMDET_VERSION="3.2.0"
NUMPY_VERSION="1.26.4"
SETUPTOOLS_VERSION="69.5.1"

TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
MMCV_FIND_LINKS="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html"

DET_CONFIG="${REPO_ROOT}/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
DET_CHECKPOINT="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CONFIG="${REPO_ROOT}/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
POSE_CHECKPOINT="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

env_exists() {
    conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

install_optional_package() {
    if ! python -m pip install "$@"; then
        echo "Warning: optional install failed: python -m pip install $*" >&2
    fi
}

require_command conda

echo "Initializing conda..."
CONDA_HOOK="$(conda shell.bash hook)"
if [[ -z "${CONDA_HOOK}" ]]; then
    echo "Failed to initialize conda shell integration." >&2
    exit 1
fi
eval "${CONDA_HOOK}"

if [[ "${KEEP_EXISTING_ENV}" -eq 0 ]] && env_exists; then
    echo "Removing existing conda environment: ${ENV_NAME}"
    conda remove -n "${ENV_NAME}" --all -y
fi

if ! env_exists; then
    echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

echo "Activating conda environment: ${ENV_NAME}"
conda activate "${ENV_NAME}"

echo "Installing packaging toolchain..."
python -m pip install -U pip
python -m pip install --force-reinstall "setuptools==${SETUPTOOLS_VERSION}" wheel packaging
python -m pip install --force-reinstall "numpy==${NUMPY_VERSION}"

echo "Installing PyTorch with CUDA 12.1..."
python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

echo "Installing OpenMMLab runtime packages..."
python -m pip install "mmengine==${MMENGINE_VERSION}"
python -m pip install "mmcv==${MMCV_VERSION}" -f "${MMCV_FIND_LINKS}"
python -m pip install "mmdet==${MMDET_VERSION}"

echo "Installing MMPose Python dependencies..."
python -m pip install json_tricks matplotlib munkres opencv-python pillow scipy xtcocotools
install_optional_package --no-build-isolation chumpy

echo "Installing current MMPose repo in editable mode..."
cd "${REPO_ROOT}"
python -m pip install -e . --no-deps

echo "Verifying imports..."
python -c "import numpy, setuptools, torch, mmcv, mmengine, mmdet, mmpose; print('numpy=' + numpy.__version__); print('setuptools=' + setuptools.__version__); print('torch=' + torch.__version__); print('cuda=' + str(torch.version.cuda)); print('gpu=' + str(torch.cuda.is_available())); print('mmcv=' + mmcv.__version__); print('mmengine=' + mmengine.__version__); print('mmdet=' + mmdet.__version__); print('mmpose=' + mmpose.__version__)"

if [[ "${SKIP_DEMO}" -eq 0 ]]; then
    echo "Running top-down demo smoke test..."
    mkdir -p "${SMOKE_TEST_OUTPUT}"

    python demo/topdown_demo_with_mmdet.py \
        "${DET_CONFIG}" \
        "${DET_CHECKPOINT}" \
        "${POSE_CONFIG}" \
        "${POSE_CHECKPOINT}" \
        --input "${SMOKE_TEST_IMAGE}" \
        --device cuda:0 \
        --output-root "${SMOKE_TEST_OUTPUT}"

    echo "Smoke test output saved to: ${SMOKE_TEST_OUTPUT}"
fi

echo "Environment setup complete."

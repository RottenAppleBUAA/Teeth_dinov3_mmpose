#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

PIPELINE="${PIPELINE:-anatomical}"
BACKBONE="${BACKBONE:-s}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-400}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-64}"
STAGE1_WORKERS="${STAGE1_WORKERS:-8}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
STAGE2_WORKERS="${STAGE2_WORKERS:-8}"

show_usage() {
  cat <<'EOF'
Usage:
  bash tools/train_panoramic_teeth_structured_dinov3_stage12.sh

Optional environment variables:
  PIPELINE=anatomical|anatomical-pointmask
  BACKBONE=s|b
  STAGE1_EPOCHS=400
  STAGE1_BATCH_SIZE=64
  STAGE1_WORKERS=8
  STAGE2_EPOCHS=100
  STAGE2_BATCH_SIZE=32
  STAGE2_WORKERS=8
  PYTHON_BIN=python

Examples:
  PIPELINE=anatomical BACKBONE=b bash tools/train_panoramic_teeth_structured_dinov3_stage12.sh
  PIPELINE=anatomical-pointmask BACKBONE=b STAGE1_BATCH_SIZE=32 STAGE2_BATCH_SIZE=16 \
    bash tools/train_panoramic_teeth_structured_dinov3_stage12.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_usage
  exit 0
fi

case "${PIPELINE}" in
  anatomical|anatomical-pointmask)
    ;;
  *)
    echo "Unsupported PIPELINE=${PIPELINE}. Expected anatomical or anatomical-pointmask." >&2
    exit 1
    ;;
esac

case "${BACKBONE}" in
  s|b)
    ;;
  *)
    echo "Unsupported BACKBONE=${BACKBONE}. Expected s or b." >&2
    exit 1
    ;;
esac

CONFIG_PREFIX="panoramic-teeth-${PIPELINE}_dinov3-convnext-${BACKBONE}_8xb32"
STAGE1_CONFIG="projects/panoramic_teeth_structured/configs/${CONFIG_PREFIX}-200e_v2-192x512_stage1.py"
STAGE2_CONFIG="projects/panoramic_teeth_structured/configs/${CONFIG_PREFIX}-50e_v2-192x512_stage2.py"

if [[ ! -f "${REPO_ROOT}/${STAGE1_CONFIG}" ]]; then
  echo "Stage1 config not found: ${STAGE1_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/${STAGE2_CONFIG}" ]]; then
  echo "Stage2 config not found: ${STAGE2_CONFIG}" >&2
  exit 1
fi

RUN_TAG="${PIPELINE}_dinov3-convnext-${BACKBONE}"
STAGE1_WORK_DIR="${REPO_ROOT}/work_dirs/${RUN_TAG}_stage1_${STAGE1_EPOCHS}e_bs${STAGE1_BATCH_SIZE}_w${STAGE1_WORKERS}"
STAGE2_WORK_DIR="${REPO_ROOT}/work_dirs/${RUN_TAG}_stage2_${STAGE2_EPOCHS}e_bs${STAGE2_BATCH_SIZE}_w${STAGE2_WORKERS}"

cd "${REPO_ROOT}"

echo "[Pipeline] ${PIPELINE}"
echo "[Backbone] ${BACKBONE}"
echo "[Stage1] train config: ${STAGE1_CONFIG}"
echo "[Stage1] work_dir: ${STAGE1_WORK_DIR}"
"${PYTHON_BIN}" tools/train.py "${STAGE1_CONFIG}" \
  --work-dir "${STAGE1_WORK_DIR}" \
  --cfg-options \
    train_cfg.max_epochs="${STAGE1_EPOCHS}" \
    train_dataloader.batch_size="${STAGE1_BATCH_SIZE}" \
    train_dataloader.num_workers="${STAGE1_WORKERS}" \
    val_dataloader.num_workers="${STAGE1_WORKERS}" \
    test_dataloader.num_workers="${STAGE1_WORKERS}"

STAGE1_BEST_CKPT=""
if [[ -f "${STAGE1_WORK_DIR}/best_NME.pth" ]]; then
  STAGE1_BEST_CKPT="${STAGE1_WORK_DIR}/best_NME.pth"
else
  STAGE1_BEST_CKPT="$(find "${STAGE1_WORK_DIR}" -maxdepth 1 -type f -name 'best_NME_epoch_*.pth' | sort -V | tail -n 1)"
fi

if [[ -z "${STAGE1_BEST_CKPT}" || ! -f "${STAGE1_BEST_CKPT}" ]]; then
  echo "Stage1 best checkpoint not found in ${STAGE1_WORK_DIR}" >&2
  exit 1
fi

echo "[Stage2] train config: ${STAGE2_CONFIG}"
echo "[Stage2] work_dir: ${STAGE2_WORK_DIR}"
echo "[Stage2] load_from: ${STAGE1_BEST_CKPT}"
"${PYTHON_BIN}" tools/train.py "${STAGE2_CONFIG}" \
  --work-dir "${STAGE2_WORK_DIR}" \
  --cfg-options \
    load_from="${STAGE1_BEST_CKPT}" \
    train_cfg.max_epochs="${STAGE2_EPOCHS}" \
    train_dataloader.batch_size="${STAGE2_BATCH_SIZE}" \
    train_dataloader.num_workers="${STAGE2_WORKERS}" \
    val_dataloader.num_workers="${STAGE2_WORKERS}" \
    test_dataloader.num_workers="${STAGE2_WORKERS}"

echo "Stage1 + Stage2 finished."

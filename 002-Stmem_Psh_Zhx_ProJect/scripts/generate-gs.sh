#!/bin/bash
# =============================================================================
# GS点云生成器 - 供stmem-psh调用
# 使用 FastGS 进行真实 3DGS 训练
# 用法: ./generate-gs.sh <batch_id> <frames_dir> <output_dir>
#
# 状态文件: <output_dir>/<batch_id>_status.json
# 输出文件: <output_dir>/<batch_id>.ply
# =============================================================================
set -e

BATCH_ID="$1"
FRAMES_DIR="$2"
OUTPUT_DIR="$3"
STATUS_FILE="${OUTPUT_DIR}/${BATCH_ID}_status.json"
PLY_OUTPUT="${OUTPUT_DIR}/${BATCH_ID}.ply"

# ========== 状态写入函数 ==========
write_status() {
  local status="$1"
  local progress="$2"
  local msg="$3"
  local ply_url="${4:-}"
  local ts=$(date -Iseconds)
  cat > "$STATUS_FILE" << STATUSEOF
{
  "batch_id": "${BATCH_ID}",
  "status": "${status}",
  "progress": ${progress},
  "message": "${msg}",
  "ply_url": "${ply_url}",
  "updated_at": "${ts}"
}
STATUSEOF
  echo "[GS-GEN] ${status} (${progress}%): ${msg}"
}

# ========== 参数检查 ==========
if [ -z "$BATCH_ID" ] || [ -z "$FRAMES_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <batch_id> <frames_dir> <output_dir>"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
write_status "starting" 0 "开始生成高斯点云..."

# ========== 确定数据目录 ==========
# $2 可以是 frames 目录，也可以是完整的数据目录（包含 frames/, poses/, point/, intrinsics.*）
if [ -d "${FRAMES_DIR}" ] && [ -f "${FRAMES_DIR}/../intrinsics.json" ]; then
  # frames_dir 指向 data/xxx/frames/，往上退一级是完整数据目录
  DATA_DIR="$(cd "${FRAMES_DIR}/.." && pwd)"
elif [ -d "${FRAMES_DIR}" ] && { [ -f "${FRAMES_DIR}/intrinsics.json" ] || [ -f "${FRAMES_DIR}/intrinsics.yaml" ]; }; then
  # frames_dir 本身就是完整数据目录
  DATA_DIR="${FRAMES_DIR}"
else
  # fallback to default jszn
  PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  DATA_DIR="${PROJECT_ROOT}/data/jszn"
fi

# ========== 检查图片数量 ==========
if [ -d "${DATA_DIR}/frames" ]; then
  FRAME_SRC="${DATA_DIR}/frames"
elif [ -d "${DATA_DIR}/rgb" ]; then
  FRAME_SRC="${DATA_DIR}/rgb"
else
  FRAME_SRC="${DATA_DIR}"
fi

IMAGE_COUNT=$(find "${FRAME_SRC}" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) ! -name 'depth_*' | wc -l)
echo "[GS-GEN] 数据目录: ${DATA_DIR}, 图片数: ${IMAGE_COUNT}"

if [ "$IMAGE_COUNT" -lt 5 ]; then
  write_status "failed" 0 "图片数量不足 (${IMAGE_COUNT} < 5)"
  exit 1
fi

# ========== FastGS 训练管线 ==========
write_status "generating" 10 "使用 FastGS 训练 3DGS..."

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/gaussian-generation"

# 检查数据目录是否存在且包含必要文件
if [ ! -d "${DATA_DIR}/rgb" ] && [ ! -d "${DATA_DIR}/frames" ]; then
  write_status "failed" 0 "数据目录不存在: ${DATA_DIR}"
  exit 1
fi

if [ ! -f "${DATA_DIR}/intrinsics.yaml" ] && [ ! -f "${DATA_DIR}/intrinsics.json" ]; then
  write_status "failed" 0 "缺少 intrinsics.yaml，请先运行数据适配"
  exit 1
fi

# 使用 conda run 直接调用 d4rt 环境的 python
CONDA_BASE="/home/sscy/miniconda3"
CONDA_PYTHON="${CONDA_BASE}/envs/d4rt/bin/python3"

if [ ! -f "${CONDA_PYTHON}" ]; then
  write_status "failed" 0 "找不到 d4rt 环境: ${CONDA_PYTHON}"
  exit 1
fi

write_status "generating" 30 "运行 FastGS 重建（约 2-3 分钟）..."

"${CONDA_PYTHON}" "${SCRIPT_DIR}/scripts/run_fastgs.py" \
  --images "${DATA_DIR}" \
  --output "${SCRIPT_DIR}/fastgs_output" \
  --ply "${PLY_OUTPUT}" \
  --mode colmap \
  --voxel_size 0.02 \
  --iterations 30000 \
  --force \
  2>&1 || {
    write_status "failed" 0 "FastGS 训练失败"
    exit 1
  }

# ========== 完成 ==========
if [ -f "$PLY_OUTPUT" ]; then
  PLY_SIZE=$(ls -lh "$PLY_OUTPUT" | awk '{print $5}')
  write_status "done" 100 "高斯点云生成完成 (${PLY_SIZE})" \
    "/api/gaussian-splats/${BATCH_ID}.ply"
  echo "[GS-GEN] 完成: ${PLY_OUTPUT} (${PLY_SIZE})"
else
  write_status "failed" 0 "训练完成但未找到输出 PLY"
  exit 1
fi
echo "[GS-GEN] 状态: ${STATUS_FILE}"

#!/bin/bash
# =============================================================================
# DYNAMIC-GSG run_all.sh - HTTP 版状态发布（融合了 3090 核心算法流水线）
# =============================================================================
# 使用 status_sender_http.py 替代 dgsg_status_publisher.py
# 前置：pip install websockets
#       先启动: python scripts/status_bridge_http.py
# =============================================================================

set -e

# ===================== 1. 环境与路径初始化 =====================
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR"

start_time=$(date +%s)
start_time_readable=$(date +"%Y-%m-%d %H:%M:%S")

source ${BASE_DIR}/miniconda3/etc/profile.d/conda.sh

echo "==================== 初始化运行环境 ===================="
echo "脚本所在目录: $SCRIPT_DIR"
echo "工作根目录: $BASE_DIR"
echo "--------------------------------------------------------"

echo "==================== 开始执行DGSG流程 ===================="
echo "脚本开始执行时间：$start_time_readable"
echo "--------------------------------------------------------"

publish_status() {
  python scripts/status_sender_http.py --step "$1" --status "$2" 2>/dev/null || true
}

# 解析 configs/mydata/dgsg.py 获取 progress.txt 路径
PROGRESS_FILE=$(python3 -c "
import os
import importlib.util
spec = importlib.util.spec_from_file_location('config', 'configs/mydata/dgsg.py')
cfg_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg_mod)
basedir = cfg_mod.config['data']['basedir']
run_name = cfg_mod.config['run_name']
path = os.path.abspath(os.path.join(basedir, run_name, 'progress.txt'))
print(path)
" | tr -d '\r')
# 确保目录存在
mkdir -p "$(dirname "$PROGRESS_FILE")"

# 如果文件存在则清空，如果不存在则创建
> "$PROGRESS_FILE"


# 创建日志目录和文件
LOG_DIR="${BASE_DIR}/Dynamic-GSG/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_all_$(date +'%Y%m%d_%H%M%S').log"

# 使用 script 命令来记录终端输出，它能保留 TTY 格式（包括颜色和进度条）
if [ -z "$SCRIPT_LOGGING_ACTIVE" ]; then
    export SCRIPT_LOGGING_ACTIVE=1
    echo "========================================================"
    echo "=== 脚本执行日志将实时保存到: $LOG_FILE ==="
    echo "========================================================"
    exec script --return --quiet --flush -c "bash \"$0\" \"$@\"" "$LOG_FILE"
fi

# ===================== 2. 核心算法流水线 =====================
STEP_COUNTER=1
STEP_TOTAL=9

# 计算百分比的辅助函数
calc_percent() {
    awk "BEGIN {printf \"%.2f\", $STEP_COUNTER/$STEP_TOTAL*100}"
}

# [前置可选步骤] 机器狗位姿与RGBD采集（当前已注释）
# echo "【预处理】获取当前机器狗的位置信息 ..."
# publish_status 0 "running"
# python get_g2_position.py configs/mydata/dgsg.py
# publish_status 0 "success"

echo " 0%【初始化】准备开始执行DGSG流程..." >> "$PROGRESS_FILE"
publish_status $STEP_COUNTER "running"
/home/orin/miniconda3/envs/dgsg/bin/python3 ./capture_rgb_depth.py configs/mydata/dgsg.py
publish_status $STEP_COUNTER "success"
STEP_COUNTER=$((STEP_COUNTER+1))


# [步骤 1] 运行深度优化脚本 (lingbot-depth)
# echo "【${STEP_COUNTER}】执行 lingbot_depth.py ..."
# publish_status $STEP_COUNTER "running"
# conda activate lingbot-depth
# cd ${BASE_DIR}/lingbot-depth
# python process_room.py --config_path ${SCRIPT_DIR}/configs/mydata/dgsg.py
# publish_status $STEP_COUNTER "success"
# STEP_COUNTER=$((STEP_COUNTER+1))

# [步骤 2] 运行位姿估计脚本 (DROID-SLAM)
echo "$(calc_percent)%【${STEP_COUNTER}】执行 pose_estimation.py ... (位姿估计)" >> "$PROGRESS_FILE"
publish_status $STEP_COUNTER "running"
conda activate droid
cd ${BASE_DIR}/DROID-SLAM
python ./pose_estimation.py --datadir ${SCRIPT_DIR}/configs/mydata/dgsg.py
publish_status $STEP_COUNTER "success"
STEP_COUNTER=$((STEP_COUNTER+1))

# [步骤 3] 运行 DGSG 算法 (3D 高斯建图与目标关联)
echo "$(calc_percent)%【${STEP_COUNTER}】执行 dynamic_gsg_real_ssim.py ... (3D 高斯建图与目标关联)" >> "$PROGRESS_FILE"
publish_status $STEP_COUNTER "running"
conda activate dgsg
cd ${SCRIPT_DIR}
python scripts/dynamic_gsg_real_ssim.py configs/mydata/dgsg.py
publish_status $STEP_COUNTER "success"
STEP_COUNTER=$((STEP_COUNTER+1))

# [步骤 4] 导出高斯点云 PLY 文件
echo "$(calc_percent)%【${STEP_COUNTER}】执行 export_ply.py ... (导出高斯点云 PLY 文件)" >> "$PROGRESS_FILE"
publish_status $STEP_COUNTER "running"
python scripts/export_ply.py configs/mydata/dgsg.py
publish_status $STEP_COUNTER "success"
STEP_COUNTER=$((STEP_COUNTER+1))

# [步骤 5] 导出并构建场景图关系
echo "$(calc_percent)%【${STEP_COUNTER}】执行 construct_scene_graph.py ... (导出并构建场景图关系)" >> "$PROGRESS_FILE"
publish_status $STEP_COUNTER "running"
python scripts/construct_scene_graph.py configs/mydata/dgsg.py
publish_status $STEP_COUNTER "success"
STEP_COUNTER=$((STEP_COUNTER+1))

# [步骤 6] 可视化场景图中心点
echo "$(calc_percent)%【${STEP_COUNTER}】执行 visualize_centers.py ... (可视化场景图中心点)" >> "$PROGRESS_FILE"
publish_status $STEP_COUNTER "running"
python scripts/visualize_centers.py configs/mydata/dgsg.py
publish_status $STEP_COUNTER "success"
STEP_COUNTER=$((STEP_COUNTER+1))

# [步骤 7] 可视化物体高斯点云
echo "$(calc_percent)%【${STEP_COUNTER}】执行 visualize_object_gaussian.py ... (可视化物体高斯点云)" >> "$PROGRESS_FILE"
publish_status $STEP_COUNTER "running"
python scripts/visualize_object_gaussians.py configs/mydata/dgsg.py
publish_status $STEP_COUNTER "success"
STEP_COUNTER=$((STEP_COUNTER+1))

# [步骤 8] 计算相机坐标到雷达坐标的变换矩阵
echo "$(calc_percent)%【${STEP_COUNTER}】执行 calculate_matrix.py ... (计算相机坐标到雷达坐标的变换矩阵)" >> "$PROGRESS_FILE"
publish_status $STEP_COUNTER "running"
python scripts/calculate_matrix.py configs/mydata/dgsg.py
publish_status $STEP_COUNTER "success"
STEP_COUNTER=$((STEP_COUNTER+1))

# [步骤 9] 转换场景图为雷达坐标
echo "$(calc_percent)%【${STEP_COUNTER}】执行 convert_sg_to_lidar.py ... (转换场景图为雷达坐标)" >> "$PROGRESS_FILE"
publish_status $STEP_COUNTER "running"
python scripts/convert_sg_to_lidar.py configs/mydata/dgsg.py
publish_status $STEP_COUNTER "success"
STEP_COUNTER=$((STEP_COUNTER+1))


# ===================== 3. 本地耗时统计 =====================
end_time=$(date +%s)
end_time_readable=$(date +"%Y-%m-%d %H:%M:%S")
total_seconds=$((end_time - start_time))
total_hours=$((total_seconds / 3600))
total_minutes=$(( (total_seconds % 3600) / 60 ))
total_remaining_seconds=$((total_seconds % 60))

echo "--------------------------------------------------------"
echo "==================== Orin 端计算完成！ ===================="
echo "总运行时间：${total_hours}小时 ${total_minutes}分钟 ${total_remaining_seconds}秒"
echo "--------------------------------------------------------"

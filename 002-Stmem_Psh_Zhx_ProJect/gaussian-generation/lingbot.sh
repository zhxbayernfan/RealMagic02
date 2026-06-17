#!/bin/bash
# 脚本功能：按顺序运行DGSG项目的三个核心Python脚本，并统计总运行时间
# 执行说明：将脚本放在项目根目录下，执行命令：bash run_dgsg_pipeline.sh

# 设置脚本执行规则：任意命令执行失败时立即退出脚本
set -e

# ===================== 定义时间格式化函数 =====================
# 输入：秒数 ($1)
# 输出：X小时 Y分钟 Z秒
format_time() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    echo "${hours}小时 ${minutes}分钟 ${seconds}秒"
}

# ===================== 时间统计初始化 =====================
# 记录脚本开始时间（时间戳，单位：秒）
start_time=$(date +%s)
# 记录脚本开始的可读时间（如：2026-02-03 15:30:00）
start_time_readable=$(date +"%Y-%m-%d %H:%M:%S")

# ===================== 执行任务并分别计时 =====================
echo "【1/4】执行 dgsg_refactor_lingbot.py ..."
t1_start=$(date +%s)
python scripts/dgsg_refactor_lingbot.py configs/mydata/lingbot.py
t1_end=$(date +%s)

echo "【2/4】执行 construct_scene_graph_lingbot.py ..."
t2_start=$(date +%s)
python scripts/construct_scene_graph_lingbot.py configs/mydata/lingbot.py
t2_end=$(date +%s)

# echo "【3/4】执行 export_ply.py ..."
# python scripts/export_ply.py configs/mydata/lingbot.py
# python scripts/export_ply.py configs/mydata/lingbot.py --mask

# echo "【4/4】执行 visualize_centers.py ..."
# python scripts/visualize_centers.py configs/mydata/lingbot.py

# ===================== 计算并打印总运行时间 =====================
# 记录脚本结束时间（时间戳）
end_time=$(date +%s)
# 记录脚本结束的可读时间
end_time_readable=$(date +"%Y-%m-%d %H:%M:%S")
# 计算总耗时（秒）
total_seconds=$((end_time - start_time))

echo "--------------------------------------------------------"
echo "==================== 所有脚本执行完成！ ===================="
echo "脚本开始执行时间：$start_time_readable"
echo "脚本结束执行时间：$end_time_readable"
echo "任务1 (建图加2d到3d关联) 运行时间: $(format_time $((t1_end - t1_start)))"
echo "任务2 (构建场景图) 运行时间: $(format_time $((t2_end - t2_start)))"
echo "总运行时间：$(format_time $total_seconds)"
echo "总运行时间（秒）：$total_seconds 秒"
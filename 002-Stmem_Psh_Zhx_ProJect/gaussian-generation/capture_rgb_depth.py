import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import shutil
from importlib.machinery import SourceFileLoader

# 配置参数
WIDTH = 640
HEIGHT = 360
FPS = 30
MAX_FRAMES = 150  # 最多获取的帧数
STRIDE = 10  # 帧间隔，每10帧保存一次

# 深度阈值配置
DEPTH_MIN = 0
DEPTH_MAX = 20

def load_config(config_path):
    """Load configuration from a python file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    experiment = SourceFileLoader(os.path.basename(config_path), config_path).load_module()
    if not hasattr(experiment, 'config'):
        raise AttributeError("Config file must contain a 'config' dictionary")

    return experiment.config


def backup_existing_data(output_dir):
    """
    如果 poses/ 目录存在，说明是二次扫描，需要备份已有数据到 first/。

    备份内容:
      rgb/ → first/rgb/
      depth/ → first/depth/
      poses/ → first/poses/
      poses.txt → first/poses.txt
    """
    poses_path = os.path.join(output_dir, "poses")
    if not os.path.exists(poses_path):
        return False

    first_dir = os.path.join(output_dir, "first")
    print(f"\n  检测到 poses/ 目录，判定为二次扫描")
    print(f"  正在备份已有数据到 first/ ...")

    # 备份 rgb/
    src_rgb = os.path.join(output_dir, "rgb")
    if os.path.exists(src_rgb):
        shutil.copytree(src_rgb, os.path.join(first_dir, "rgb"))
        print(f"    rgb/ -> first/rgb/ ({len(os.listdir(src_rgb))} 帧)")

    # 备份 depth/
    src_depth = os.path.join(output_dir, "depth")
    if os.path.exists(src_depth):
        shutil.copytree(src_depth, os.path.join(first_dir, "depth"))
        print(f"    depth/ -> first/depth/ ({len(os.listdir(src_depth))} 帧)")

    # 备份 poses/
    shutil.copytree(poses_path, os.path.join(first_dir, "poses"))
    print(f"    poses/ -> first/poses/ ({len(os.listdir(poses_path))} 帧)")

    # 备份 poses.txt（如果有）
    poses_txt = os.path.join(output_dir, "poses.txt")
    if os.path.exists(poses_txt):
        shutil.copy2(poses_txt, os.path.join(first_dir, "poses.txt"))
        print(f"    poses.txt -> first/poses.txt")

    # 清理旧数据，为新采集腾出空间
    shutil.rmtree(src_rgb, ignore_errors=True)
    shutil.rmtree(src_depth, ignore_errors=True)
    shutil.rmtree(poses_path, ignore_errors=True)
    if os.path.exists(os.path.join(output_dir, "poses.txt")):
        os.remove(os.path.join(output_dir, "poses.txt"))
    print(f"  旧数据已清理，可以开始新采集\n")
    return True


def create_output_folder(config):
    """根据配置创建输出文件夹，如果是二次扫描则自动备份旧数据"""

    basedir = config['data'].get('basedir', './data/mydata')
    run_name = config.get('run_name', 'default_run')

    output_dir = os.path.join(basedir, run_name)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 二次扫描检测与备份
    backup_existing_data(output_dir)

    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print(f"Output directories created at:\n  RGB: {rgb_dir}\n  Depth: {depth_dir}")

    return output_dir, rgb_dir, depth_dir


def _init_depth_process():
    """初始化深度处理流程，设置各种滤波器以提高深度数据质量"""
    # 创建深度到视差的转换器（True表示转换到视差空间）
    depth_to_disparity = rs.disparity_transform(True)
    # 创建视差到深度的转换器（False表示转换回深度空间）
    disparity_to_depth = rs.disparity_transform(False)
    # 创建空间滤波器，用于平滑深度图像
    spatial = rs.spatial_filter()
    # 设置空间滤波器的幅度参数（控制滤波强度）
    spatial.set_option(rs.option.filter_magnitude, 5)
    # 设置空间滤波器的平滑alpha参数（控制平滑程度，0-1之间）
    spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
    # 设置空间滤波器的平滑delta参数（控制平滑的阈值）
    spatial.set_option(rs.option.filter_smooth_delta, 1)
    # 设置空间滤波器的孔洞填充参数（1=轻度填充，3=激进填充）
    # 权衡：fill=0 不产生边缘涂抹但边缘零值高（>15%），fill=1 不影响 DGS/3DGS 质量的同时可显著降低零值比例
    spatial.set_option(rs.option.holes_fill, 1)
    # 创建时间滤波器，用于在时间维度上平滑深度数据
    temporal = rs.temporal_filter()
    # 设置时间滤波器的平滑alpha参数
    temporal.set_option(rs.option.filter_smooth_alpha, 0.75)
    # 设置时间滤波器的平滑delta参数
    temporal.set_option(rs.option.filter_smooth_delta, 1)

    return depth_to_disparity, disparity_to_depth, spatial, temporal

def _process_depth(depth_frame, depth_to_disparity, disparity_to_depth, spatial, temporal):
    """对深度帧进行滤波处理，提高深度数据质量"""
    # 深度处理流程
    # 第一步：将深度转换为视差（视差空间更适合滤波）
    filtered_depth = depth_to_disparity.process(depth_frame)
    # 第二步：应用空间滤波器，平滑视差数据
    filtered_depth = spatial.process(filtered_depth)
    # 第三步：应用时间滤波器，在时间维度上平滑数据
    filtered_depth = temporal.process(filtered_depth)
    # 第四步：将视差转换回深度
    filtered_depth = disparity_to_depth.process(filtered_depth)
    # 返回处理后的深度帧
    return filtered_depth

def main():
    parser = argparse.ArgumentParser(description="Capture RGB and Depth frames based on config")
    parser.add_argument("config_path", type=str, help="Path to the configuration file (e.g., configs/mydata/dgsg.py)")
    args = parser.parse_args()

    try:
        config = load_config(args.config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # 创建输出文件夹
    output_dir, frames_dir, depths_dir = create_output_folder(config)

    print(f"分辨率: {WIDTH}x{HEIGHT}, 帧率: {FPS}")
    print(f"将获取 {MAX_FRAMES} 帧数据...")

    # 配置 RealSense 管道
    pipeline = rs.pipeline()
    rs_config = rs.config()

    # 启用深度流和彩色流
    rs_config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    rs_config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    # 启动管道
    try:
        profile = pipeline.start(rs_config)
    except RuntimeError as e:
        print(f"Error starting pipeline: {e}")
        print("Please check if the RealSense camera is connected.")
        return

    # 获取深度传感器的深度缩放因子
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # [优化1] 提升深度传感器性能：如果支持，开启最大激光发射功率以获取更致密、低噪的室内深度图
    if depth_sensor.supports(rs.option.laser_power):
        laser_range = depth_sensor.get_option_range(rs.option.laser_power)
        depth_sensor.set_option(rs.option.laser_power, laser_range.max)

    # 初始化深度处理流程
    depth_to_disparity, disparity_to_depth, spatial, temporal = _init_depth_process()

    # 获取对齐流的配置，用于将深度图对齐到彩色图
    align_to = rs.stream.color
    align = rs.align(align_to)

    # [优化2] 获取并保存对齐后的真实相机内参到 YAML 格式
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    intrinsics_path = os.path.join(output_dir, "intrinsics.yaml")
    png_depth_scale = 1.0 / depth_scale  # PNG 存原始 uint16 值，depth_scale 的倒数即为米单位的除数
    with open(intrinsics_path, "w") as f:
        f.write(f"dataset_name: 'mydata'\n")
        f.write(f"camera_params:\n")
        f.write(f"  image_height: {intrinsics.height}\n")
        f.write(f"  image_width: {intrinsics.width}\n")
        f.write(f"  fx: {intrinsics.fx}\n")
        f.write(f"  fy: {intrinsics.fy}\n")
        f.write(f"  cx: {intrinsics.ppx}\n")
        f.write(f"  cy: {intrinsics.ppy}\n")
        f.write(f"  png_depth_scale: {png_depth_scale}\n")
        f.write(f"  crop_edge: 0\n")
    print(f"已自动保存相机内参至: {intrinsics_path}")
    print(f"相机内参:")
    print(f"  fx={intrinsics.fx:.6f}, fy={intrinsics.fy:.6f}, cx={intrinsics.ppx:.6f}, cy={intrinsics.ppy:.6f}")
    print(f"  width={intrinsics.width}, height={intrinsics.height}, png_depth_scale={png_depth_scale}")

    try:
        frame_count = 0
        save_count = 0

        # 预热相机（等待 AE/AWB 稳定，100帧≈3秒足够）
        print("Warming up camera...")
        for _ in range(100):
            pipeline.wait_for_frames()

        print("Start capturing...")

        while save_count < MAX_FRAMES:
            # 等待帧
            frames = pipeline.wait_for_frames()

            # 先对齐，因为 python wrapper 只能对原始 frameset 进行对齐
            aligned_frames = align.process(frames)

            # 获取对齐后的深度帧和彩色帧
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # 对对齐后的深度帧进行滤波处理（空间滤波已关闭孔洞填充，避免边缘涂抹）
            filtered_depth_frame = _process_depth(aligned_depth_frame, depth_to_disparity, disparity_to_depth, spatial, temporal)

            # 转换为 NumPy 数组
            depth_image = np.asanyarray(filtered_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if frame_count % STRIDE == 0:
                # [优化3] 利用脚本头部定义的 DEPTH_MIN 和 DEPTH_MAX 进行深度截断，过滤背景噪声和无效过近点
                depth_in_meters = depth_image * depth_scale
                depth_image[(depth_in_meters < DEPTH_MIN) | (depth_in_meters > DEPTH_MAX)] = 0

                # 保存深度图 (png格式) - 保持为16位深度图，但这里为了兼容性还是存为png
                # 注意：cv2.imwrite保存16位图像时，depth_image必须是uint16
                # RealSense depth是Z16格式，即uint16，单位通常是毫米
                depth_filename = os.path.join(depths_dir, f'{save_count:05d}.png')
                cv2.imwrite(depth_filename, depth_image)

                # [优化4] 保存 RGB 图（修改为无损PNG格式，避免JPEG压缩伪影降低3DGS渲染质量）
                color_filename = os.path.join(frames_dir, f'{save_count:05d}.png')
                cv2.imwrite(color_filename, color_image)

                save_count += 1
                print(f"已保存第 {save_count}/{MAX_FRAMES} 帧")

            frame_count += 1

        print(f"\n完成! 共保存 {save_count} 帧数据到: {output_dir}")

    finally:
        pipeline.stop()

if __name__ == '__main__':
    main()

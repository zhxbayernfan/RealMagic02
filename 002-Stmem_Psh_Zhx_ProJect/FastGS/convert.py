#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil
import yaml

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--sequential", action='store_true', help="Use sequential matcher (for ordered video frames)")
parser.add_argument("--intrinsics", default=None, type=str,
                    help="Path to intrinsics.yaml. If set, COLMAP uses these intrinsics and fixes them during BA.")
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

# Load custom intrinsics if provided
camera_params_str = ""
mapper_fix_intrinsics = ""
if args.intrinsics:
    with open(args.intrinsics, 'r') as f:
        cfg = yaml.safe_load(f)
    cam = cfg['camera_params']
    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
    camera_params_str = " --ImageReader.camera_params {},{},{},{}".format(fx, fy, cx, cy)
    mapper_fix_intrinsics = ""
    print(f"Using intrinsics prior from {args.intrinsics}: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.1f}, cy={cy:.1f}")

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu) + camera_params_str
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    matcher = "sequential_matcher" if args.sequential else "exhaustive_matcher"
    feat_matching_cmd = colmap_command + " " + matcher + " \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse" + mapper_fix_intrinsics)
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")

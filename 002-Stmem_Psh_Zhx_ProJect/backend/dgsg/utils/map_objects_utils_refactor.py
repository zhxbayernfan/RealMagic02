from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F 
import numpy as np
import supervision as sv
import cv2
import logging
import faiss
import pickle
import gzip
import os
import open3d as o3d
import base64
import json
import time
import openai

from dam import DescribeAnythingModel
from tqdm import tqdm
from PIL import Image
from collections import Counter
from pathlib import Path
from scipy.ndimage import binary_erosion
from ultralytics import YOLO, SAM
from utils.slam_external import build_rotation
from utils.slam_classes import MapObjectList, DetectionList
from utils.slam_helpers import (
    transform_to_frame,
    transformed_params_for_detection
)
# from fast_gauss import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms, ops
from ram import inference_ram as inference
from vlm_utils.llava_local import LocalVLM


def read_color_book(color_book_path: str):
    color_book = []

    with open(color_book_path, 'r') as file:
        for line in file.readlines()[1:]:  # 跳过表头
            r, g, b = map(float, line.strip().split())
            color_book.append([r/255, g/255, b/255])

    return color_book


def compute_clip_features_batched(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    
    image = Image.fromarray(image)
    padding = 20  # Adjust the padding amount as needed
    
    image_crops = []
    preprocessed_images = []
    text_tokens = []
    
    # Prepare data for batch processing
    for idx in range(len(detections.xyxy)):
        x_min, y_min, x_max, y_max = detections.xyxy[idx]
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)

        class_id = detections.class_id[idx]
        text_tokens.append(classes[class_id])
        image_crops.append(cropped_image)
    
    # Convert lists to batches  B 3 224 224
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)   
    text_tokens_batch = clip_tokenizer(text_tokens).to(device)
    
    # Batch inference
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # text_features = clip_model.encode_text(text_tokens_batch)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy
    image_feats = image_features.cpu().numpy()
    # text_feats = text_features.cpu().numpy()
    # image_feats = []
    text_feats = []
    
    return image_crops, image_feats, text_feats


def resize_gobs(gobs, image):

    # If the shapes are the same, no resizing is necessary
    if gobs['mask'].shape[1:] == image.shape[:2]:
        return gobs

    new_masks = []

    for mask_idx in range(len(gobs['xyxy'])):

        mask = gobs['mask'][mask_idx]
        # Rescale the xyxy coordinates to the image shape
        x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
        x1 = round(x1 * image.shape[1] / mask.shape[1])
        y1 = round(y1 * image.shape[0] / mask.shape[0])
        x2 = round(x2 * image.shape[1] / mask.shape[1])
        y2 = round(y2 * image.shape[0] / mask.shape[0])
        gobs['xyxy'][mask_idx] = [x1, y1, x2, y2]

        # Reshape the mask to the image shape
        mask = cv2.resize(mask.astype(np.uint8), image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
        new_masks.append(mask)

    if len(new_masks) > 0:
        gobs['mask'] = np.asarray(new_masks)

    return gobs


def filter_gobs(
    gobs: dict,
    image: np.ndarray,
    skip_bg: bool = None,  # Explicitly passing skip_bg
    BG_CLASSES: list = None,  # Explicitly passing BG_CLASSES
    mask_area_threshold: float = 10,  # Default value as fallback
    max_bbox_area_ratio: float = None,  # Explicitly passing max_bbox_area_ratio
    mask_conf_threshold: float = None,  # Explicitly passing mask_conf_threshold
):
    # If no detection at all
    if len(gobs['xyxy']) == 0:
        return gobs

    # Filter out the objects based on various criteria
    idx_to_keep = []
    for mask_idx in range(len(gobs['xyxy'])):
        local_class_id = gobs['class_id'][mask_idx]
        class_name = gobs['classes'][local_class_id]

        # Skip masks that are too small
        mask_area = gobs['mask'][mask_idx].sum()
        if mask_area < max(mask_area_threshold, 10):
            logging.debug(f"Skipped due to small mask area ({mask_area} pixels) - Class: {class_name}")
            continue

        # Skip the BG classes
        if skip_bg and class_name in BG_CLASSES:
            logging.debug(f"Skipped background class: {class_name}")
            continue

        # Skip the non-background boxes that are too large
        if class_name not in BG_CLASSES:
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            if max_bbox_area_ratio is not None and bbox_area > max_bbox_area_ratio * image_area:
                logging.debug(f"Skipped due to large bounding box area ratio - Class: {class_name}, Area Ratio: {bbox_area/image_area:.4f}")
                continue

        # Skip masks with low confidence
        if mask_conf_threshold is not None and gobs['confidence'] is not None:
            if gobs['confidence'][mask_idx] < mask_conf_threshold:
                logging.debug(f"Skipped due to low confidence ({gobs['confidence'][mask_idx]}) - Class: {class_name}")
                continue

        idx_to_keep.append(mask_idx)

    # for key in gobs.keys():
    #     print(key, type(gobs[key]), len(gobs[key]))

    for k in gobs.keys():
        if isinstance(gobs[k], str) or k == "classes":  # Captions
            continue
        if k in ['labels', 'edges', 'detection_class_labels', 'text_feats']:
            continue
        elif isinstance(gobs[k], list):
            gobs[k] = [gobs[k][i] for i in idx_to_keep]
        elif isinstance(gobs[k], np.ndarray):
            gobs[k] = gobs[k][idx_to_keep]
        else:
            raise NotImplementedError(f"Unhandled type {type(gobs[k])}")

    return gobs


def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    '''
    Compute the containing relationship between all pair of bounding boxes.
    For each mask, subtract the mask of bounding boxes that are contained by it.
     
    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2
        
    Returns:
        mask_sub: (N, H, W), binary mask
    '''
    N = xyxy.shape[0] # number of boxes

    # Get areas of each xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) # (N,)

    # Compute intersection boxes
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)
    
    inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)  交集不为负数

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1] # (N, N)
    
    inter_over_box1 = inter_areas / areas[:, None] # (N, N)
    # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
    inter_over_box2 = inter_over_box1.T # (N, N)
    
    # if the intersection area is smaller than th2 of the area of box1, 
    # and the intersection area is larger than th1 of the area of box2,
    # then box2 is considered contained by box1
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1) # (N, N)
    contained_idx = contained.nonzero() # (num_contained, 2)

    mask_sub = mask.copy() # (N, H, W)
    # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])

    return mask_sub

def mask_erosion(masks: np.ndarray):
    num = masks.shape[0]

    for i in range(num):
        masks[i, :, :] = binary_erosion(masks[i, :, :], structure=np.ones((2, 2)))

    return masks


def merge_first_objects_itself(objects: MapObjectList, cfg):
    '''
    TODO:
        完善对mapobject的自身合并
    '''
    # print("Before merging:", len(objects))
    len_a = len(objects)
    iou_similarities = compute_mask_iou_similarities(objects, objects)
    overlap_matrix = np.zeros((len_a, len_a))

    for idx_a in range(len_a):
        for idx_b in range(len_a):
            if idx_a == idx_b:
                continue

            if iou_similarities[idx_a, idx_b] < 1e-6:
                continue

            overlap_matrix[idx_a, idx_b] = iou_similarities[idx_a, idx_b]

    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]

    kept_objects = np.ones(len(objects), dtype=bool)

    # Sort indices of overlap ratios in descending order
    sort = np.argsort(overlap_ratio)[::-1]  
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]
    
    for i, j, ratio in zip(x, y, overlap_ratio):
        if ratio > cfg['merge_overlap_thresh']:
            visual_sim = F.cosine_similarity(torch.from_numpy(objects[i]["clip_ft"]), 
                                             torch.from_numpy(objects[j]["clip_ft"]), 
                                             dim=0)

            if (visual_sim > cfg['merge_visual_sim_thresh']) :
                if kept_objects[j]:  # Check if the target object has not been merged into another
                    # Merge object i into object j
                    objects[j] = merge_obj2_into_obj1(
                        objects[j],
                        objects[i],
                    )
                    kept_objects[i] = False  # Mark object i as 'merged'

        else:
            break  # Stop processing if the current overlap ratio is below the threshold

    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)

    return objects


def merge_curr_detection_itself(objects: DetectionList, cfg):
    '''
    TODO:
        完善对mapobject的自身合并
    '''
    # print("Before merging:", len(objects))
    len_a = len(objects)
    iou_similarities = compute_mask_iou_similarities(objects, objects)
    overlap_matrix = np.zeros((len_a, len_a))

    for idx_a in range(len_a):
        for idx_b in range(len_a):
            if idx_a == idx_b:
                continue

            if iou_similarities[idx_a, idx_b] < 1e-6:
                continue

            overlap_matrix[idx_a, idx_b] = iou_similarities[idx_a, idx_b]

    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]

    kept_objects = np.ones(len(objects), dtype=bool)

    # Sort indices of overlap ratios in descending order
    sort = np.argsort(overlap_ratio)[::-1]  
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]
    
    for i, j, ratio in zip(x, y, overlap_ratio):
        if ratio > cfg['merge_overlap_thresh']:
            visual_sim = F.cosine_similarity(torch.from_numpy(objects[i]["clip_ft"]), 
                                             torch.from_numpy(objects[j]["clip_ft"]), 
                                             dim=0)

            if (visual_sim > cfg['merge_visual_sim_thresh']) :
                if kept_objects[j]:  # Check if the target object has not been merged into another
                    # Merge object i into object j
                    objects[j] = merge_obj2_into_obj1(
                        objects[j],
                        objects[i],
                    )
                    kept_objects[i] = False  # Mark object i as 'merged'

        else:
            break  # Stop processing if the current overlap ratio is below the threshold

    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = DetectionList(new_objects)

    return objects

def encode_img(image_input):

    if isinstance(image_input, str):
        # 处理文件路径
        with open(image_input, "rb") as file:
            encoded_image = base64.b64encode(file.read()).decode('utf-8')
        return f"data:image;base64,{encoded_image}"
    
    elif isinstance(image_input, np.ndarray):
        # 处理 NumPy 数组
        # 确保图像是 uint8 类型
        if image_input.dtype != np.uint8:
            image_input = image_input.astype(np.uint8)
        
        # 自动检测并转换通道顺序 (OpenCV 使用 BGR，但网页需要 RGB)
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            # 如果是 3 通道图像，转换为 RGB 格式
            image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        
        # 编码为 PNG 格式 (可改为 JPEG 如果需要更小体积)
        success, buffer = cv2.imencode('.png', image_input)
        
        if not success:
            raise ValueError("图像编码失败")
        
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return f"data:image;base64,{encoded_image}"
    
    else:
        raise TypeError("输入必须是文件路径(str)或NumPy数组")


def initialize_first_timestep_gaussian_classes(
        color: torch.Tensor,
        detection_model,
        ram_model,
        ai_client,
        sam_predictor: SAM, 
        clip_model, 
        clip_preprocess, 
        clip_tokenizer,
        obj_classes,
        color_book,
        cfg
):
    objects = MapObjectList()
    image = color.cpu().numpy().astype(np.uint8)

    if cfg['detection_model'] == 'groundingdino':
        # img_url = encode_img(image)  # Encode the first image to Base64 format
        # completion = ai_client.chat.completions.create(
        #     model="qwen2.5-vl-72b-instruct",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        #     messages=[{"role": "user","content": [
        #             {"type": "image_url",
        #             "image_url": {"url": f"{img_url}"}},
        #             {"type": "text", "text": "Describe visible object categories in front of you. Only provide the category names, no descriptions needed and each category name is separated by the character \".\""},
        #             ]}]
        #     )
        # text_prompt = completion.choices[0].message.content

        image_pil = Image.fromarray(image)
        raw_image = image_pil.resize((384, 384))
        tagging_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        raw_image = tagging_transform(raw_image).unsqueeze(0).to("cuda:0")
        ram_model.eval()
        text_prompt = inference(raw_image, ram_model)[0].replace(' | ', '.')
        classes = process_tag_classes(text_prompt=text_prompt)

        detections_gd = detection_model.predict_with_classes(    
                image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR), # This function expects a BGR image...
                classes=classes,
                box_threshold=0.3,
                text_threshold=0.3
            )
        
        if len(detections_gd.class_id) > 0:
            ### Non-maximum suppression ###
            print(f"Before NMS: {len(detections_gd.xyxy)} boxes")
            nms_idx = ops.nms(
                torch.from_numpy(detections_gd.xyxy), 
                torch.from_numpy(detections_gd.confidence), 
                0.5
            ).numpy().tolist()
            print(f"After NMS: {len(detections_gd.xyxy)} boxes")
            detections_gd.xyxy = detections_gd.xyxy[nms_idx]
            detections_gd.confidence = detections_gd.confidence[nms_idx]
            detections_gd.class_id = detections_gd.class_id[nms_idx]
            
            # Somehow some detections will have class_id=-1, remove them
            # valid_idx = detections.class_id != -1
            valid_idx = [i for i, val in enumerate(detections_gd.class_id) if (val is not None and val != -1)]
            detections_gd.xyxy = detections_gd.xyxy[valid_idx]
            detections_gd.confidence = detections_gd.confidence[valid_idx]
            detections_gd.class_id = detections_gd.class_id[valid_idx]
            
            confidences = detections_gd.confidence
            detection_class_ids = detections_gd.class_id.astype(int)
            xyxy_np = detections_gd.xyxy
            xyxy_tensor = torch.from_numpy(xyxy_np).to("cuda:0")
    else:
        results = detection_model.predict(image, conf=0.1, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        # detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
        detection_class_labels = [{obj_classes.get_classes_arr()[class_id]} for class_idx, class_id in enumerate(detection_class_ids)]
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

    # if there are detections,
    # Get Masks Using SAM or MobileSAM
    # UltraLytics SAM
    if xyxy_tensor.numel() != 0:
        sam_out = sam_predictor.predict(image, bboxes=xyxy_tensor, verbose=False)
        masks_tensor = sam_out[0].masks.data
        masks_np = masks_tensor.cpu().numpy()

    # Create a detections object that we will save later
    curr_det = sv.Detections(
        xyxy=xyxy_np,
        confidence=confidences,
        class_id=detection_class_ids,
        mask=masks_np,
    )
    
    image_crops, image_feats, text_feats = compute_clip_features_batched(
                image, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), device='cuda')
    

    results = {
        # add new uuid for each detection 
        "xyxy": curr_det.xyxy,
        "confidence": curr_det.confidence,
        "class_id": curr_det.class_id,
        "mask": curr_det.mask,
        "classes": obj_classes.get_classes_arr(),
        "image_crops": image_crops,
        "image_feats": image_feats,
        "text_feats": text_feats,
        # "detection_class_labels": detection_class_labels,
        # "labels": labels,
        # "edges": edges,
    }

    resized_gobs = resize_gobs(results, image)

    filtered_gobs = filter_gobs(resized_gobs, image, 
        skip_bg=cfg['skip_bg'],
        BG_CLASSES=obj_classes.get_bg_classes_arr(),
        mask_area_threshold=cfg['mask_area_threshold'],
        max_bbox_area_ratio=cfg['max_bbox_area_ratio'],
        mask_conf_threshold=cfg['mask_conf_threshold'],
    )

    filtered_gobs['mask'] = mask_subtract_contained(filtered_gobs['xyxy'], filtered_gobs['mask'])

    filtered_gobs['mask'] = mask_erosion(filtered_gobs['mask'])

    for idx in range(len(filtered_gobs['mask'])):
        map_object = {
            'idx' : idx + 1,
            'class_id' : [filtered_gobs['class_id'][idx]],
            'class_label' : detection_class_labels[idx],
            # 'class_label' : det.class_label,
            'mask' : filtered_gobs['mask'][idx],
            'mask_area' : filtered_gobs['mask'][idx].sum(),
            'clip_ft' : filtered_gobs['image_feats'][idx],
            'image_crops' : filtered_gobs['image_crops'][idx],
            'num_detections' : 1,  # Number of detections for this object
            'best_view' : 0
        }
        objects.append(map_object)

    objects = merge_first_objects_itself(objects, cfg) # 这步操作是为了解决 “同一物理物体被检测 / 分割成多个独立对象” 的问题（比如一张桌子被拆成 2 个掩码，或相邻的同类别椅子被识别为两个对象）。
    idx_mask = np.zeros((color.shape[0], color.shape[1]), dtype=np.int8) 
    # 构建一个表示特征颜色的tensor
    feature_mask = torch.zeros((color.shape[0], color.shape[1], 3), dtype=torch.float32)

    for i in range(len(objects)):
        mask = objects[i]['mask'] > 0
        idx_mask[mask] = objects[i]['idx']
        feature_mask[mask] = torch.tensor(color_book[objects[i]['idx']], dtype=torch.float32)
        objects[i]['visibility'] = objects[i]['mask_area'] / (color.shape[0] * color.shape[1])
        objects[i]['best_view'] = 0
        # debug   
        # mask_pixel_count = objects[i]['mask'].sum().item()

        # plt.title(f"Boolean Mask:{objects[i]['idx']}")
        # plt.imshow(objects[i]['mask'], cmap='gray')  
        # plt.axis('off')
        # plt.show()

    return idx_mask, feature_mask, objects

def process_tag_classes(text_prompt:str) -> list[str]:
    '''Convert a text prompt from Tag2Text to a list of classes. '''
    classes = text_prompt.split('.')
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != '']
    add_classes = ["picture","handle", 'Bottled Coke', 'Canned Beer', 'apple', 'potato', 'green toy', 'blue bottle', 'green container', 'blue and grey umbrella', 'blue toy', 'pen', 'orange', 'eggplant', 'yellow bottle', 'corn', 'chili pepper', 'small scissors', 'keys', 'green container', 'cabinet', 'long table', 'big table', 'plate']
    remove_classes = [
        "room", "kitchen", "office", "house", "home", "building", "corner",
        "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
        "apartment", "image", "city", "blue", "skylight", "hallway", 
        "bureau", "modern", "salon", "doorway", "wall lamp", "wood floor",
        "floor", "ladder", "sink", "counter top", "hardwood",
        "shower curtain", "curtain", "slide", "peak", "closet",
        "man", "woman", "child", "boy", "girl", "person", "human", "drawer"]
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    return classes


def calculate_view_score(mask_area, xyxy, img_width, img_height):
    """
    计算当前视角的综合得分，用于判定 best_view。
    结合了面积占比惩罚和边界框中心度。
    """
    # 1. 面积得分
    area_ratio = mask_area / (img_width * img_height)
    
    # 如果面积占比过大（>60%），说明可能“贴脸”或截断严重，给予极低分惩罚
    if area_ratio > 0.6:
        area_score = 0.1
    # 如果面积占比极小（<1%），说明太远看不清，给予一定惩罚
    elif area_ratio < 0.01:
        area_score = area_ratio * 10  # 稍微提权，但整体仍偏低
    else:
        # 在 1% ~ 60% 之间，面积越大越好，但做了平滑处理，避免无脑选最大
        area_score = area_ratio

    # 2. 中心度得分
    x_min, y_min, x_max, y_max = xyxy
    bbox_center_x = (x_min + x_max) / 2.0
    bbox_center_y = (y_min + y_max) / 2.0
    img_center_x = img_width / 2.0
    img_center_y = img_height / 2.0

    # 归一化距离惩罚（越偏离中心，距离越大）
    dist_x = abs(bbox_center_x - img_center_x) / img_center_x
    dist_y = abs(bbox_center_y - img_center_y) / img_center_y
    
    # 距离越小，中心度越高，范围在 [0, 1] 之间
    center_score = 1.0 - max(dist_x, dist_y)
    
    # 3. 边界截断惩罚（可选：如果 BBox 紧贴图像边缘，说明物体可能不完整）
    edge_margin = 10
    edge_penalty = 1.0
    if x_min < edge_margin or y_min < edge_margin or x_max > (img_width - edge_margin) or y_max > (img_height - edge_margin):
        edge_penalty = 0.5  # 触碰边缘，得分减半

    return area_score * center_score * edge_penalty


def process_this_frame_detection(
        image: torch.Tensor, 
        time_idx: int,
        detection_model, 
        ram_model,
        ai_client,
        sam_predictor: SAM, 
        clip_model, 
        clip_preprocess, 
        clip_tokenizer,
        obj_classes,
        cfg
)-> list:
    # Initialize timers
    global sam_time_sum, yolo_time_sum, clip_time_sum
    if 'sam_time_sum' not in globals():
        sam_time_sum = 0
        yolo_time_sum = 0
        clip_time_sum = 0
        
    detections = []
    image = image.cpu().numpy().astype(np.uint8)

    yolo_start = time.time()
    if cfg['detection_model'] == 'groundingdino':
        # img_url = encode_img(image)  # Encode the first image to Base64 format
        # completion = ai_client.chat.completions.create(
        #     model="qwen2.5-vl-72b-instruct",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        #     messages=[{"role": "user","content": [
        #             {"type": "image_url",
        #             "image_url": {"url": f"{img_url}"}},
        #             {"type": "text", "text": "Describe visible object categories in front of you. Only provide the category names, no descriptions needed and each category name is separated by the character \".\""},
        #             ]}]
        #     )
        # text_prompt = completion.choices[0].message.content
        image_pil = Image.fromarray(image)
        raw_image = image_pil.resize((384, 384))
        tagging_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        raw_image = tagging_transform(raw_image).unsqueeze(0).to("cuda:0")
        ram_model.eval()
        text_prompt = inference(raw_image, ram_model)[0].replace(' | ', '.')
        classes = process_tag_classes(text_prompt=text_prompt)

        detections_gd = detection_model.predict_with_classes(    
                image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR), # This function expects a BGR image...
                classes=classes,
                box_threshold=0.3,
                text_threshold=0.3
            )
        
        if len(detections_gd.class_id) > 0:
            ### Non-maximum suppression ###
            print(f"Before NMS: {len(detections_gd.xyxy)} boxes")
            nms_idx = ops.nms(
                torch.from_numpy(detections_gd.xyxy), 
                torch.from_numpy(detections_gd.confidence), 
                0.5
            ).numpy().tolist()
            print(f"After NMS: {len(detections_gd.xyxy)} boxes")
            detections_gd.xyxy = detections_gd.xyxy[nms_idx]
            detections_gd.confidence = detections_gd.confidence[nms_idx]
            detections_gd.class_id = detections_gd.class_id[nms_idx]
            
            # Somehow some detections will have class_id=-1, remove them
            # valid_idx = detections.class_id != -1
            valid_idx = [i for i, val in enumerate(detections_gd.class_id) if (val is not None and val != -1)]
            detections_gd.xyxy = detections_gd.xyxy[valid_idx]
            detections_gd.confidence = detections_gd.confidence[valid_idx]
            detections_gd.class_id = detections_gd.class_id[valid_idx]
            
            confidences = detections_gd.confidence
            detection_class_ids = detections_gd.class_id.astype(int)
            xyxy_np = detections_gd.xyxy
            xyxy_tensor = torch.from_numpy(xyxy_np).to("cuda:0")
    else:
        results = detection_model.predict(image, conf=0.1, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        # detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
        detection_class_labels = [{obj_classes.get_classes_arr()[class_id]} for class_idx, class_id in enumerate(detection_class_ids)]
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()
    yolo_end = time.time()
    yolo_time_sum += (yolo_end - yolo_start)

    sam_start = time.time()
    if xyxy_tensor.numel() != 0:
        sam_out = sam_predictor.predict(image, bboxes=xyxy_tensor, verbose=False)
        masks_tensor = sam_out[0].masks.data
        masks_np = masks_tensor.cpu().numpy()
    else:
        sam_end = time.time()
        sam_time_sum += (sam_end - sam_start)
        return detections
    sam_end = time.time()
    sam_time_sum += (sam_end - sam_start)

    # curr_det = sv.Detections(
    #     xyxy=xyxy_np,
    #     confidence=confidences,  # np
    #     class_id=detection_class_ids,  # np
    #     mask=masks_np,
    # )
    
    # 转换为 dict 结构，移除 sv.Detections 依赖，简化数据流
    curr_det_dict = {
        "xyxy": xyxy_np,
        "confidence": confidences,
        "class_id": detection_class_ids,
        "mask": masks_np,
    }

    # 为了适配 compute_clip_features_batched 临时封装一个简单的类
    class SimpleDetections:
        def __init__(self, xyxy, class_id):
            self.xyxy = xyxy
            self.class_id = class_id
    
    simple_det = SimpleDetections(xyxy_np, detection_class_ids)

    clip_start = time.time()
    image_crops, image_feats, text_feats = compute_clip_features_batched(
                                                image, simple_det, 
                                                clip_model, clip_preprocess, clip_tokenizer, 
                                                obj_classes.get_classes_arr(), device='cuda')
    clip_end = time.time()
    clip_time_sum += (clip_end - clip_start)

    print(f"Avg Time - YOLO/GD: {yolo_time_sum/(time_idx+1):.4f}s, SAM: {sam_time_sum/(time_idx+1):.4f}s, CLIP: {clip_time_sum/(time_idx+1):.4f}s")
    
    results = {
        "xyxy": curr_det_dict["xyxy"],
        "confidence": curr_det_dict["confidence"],
        "class_id": curr_det_dict["class_id"],
        "mask": curr_det_dict["mask"],
        "classes": obj_classes.get_classes_arr(),
        "image_crops": image_crops,
        "image_feats": image_feats,
        "text_feats": text_feats,
    }

    resized_gobs = resize_gobs(results, image)

    filtered_gobs = filter_gobs(resized_gobs, image, 
        skip_bg=True,
        BG_CLASSES=obj_classes.get_bg_classes_arr(),
        mask_area_threshold=cfg['mask_area_threshold'],
        max_bbox_area_ratio=cfg['max_bbox_area_ratio'],
        mask_conf_threshold=cfg['mask_conf_threshold'],
    )

    filtered_gobs['mask'] = mask_subtract_contained(filtered_gobs['xyxy'], filtered_gobs['mask'])

    # Get image dimensions for view scoring (using 'image' instead of 'rgb_image')
    img_height, img_width = image.shape[:2]

    for idx in range(len(filtered_gobs['mask'])):
        mask_area = filtered_gobs['mask'][idx].sum()
        num_labels, _ = cv2.connectedComponents(filtered_gobs['mask'][idx].astype(np.uint8), connectivity=4)
        if mask_area > 500 and (num_labels - 1) < 5 :
            filtered_gobs['mask'][idx] = binary_erosion(filtered_gobs['mask'][idx], structure=np.ones((3, 3)))
            
            # Calculate view score
            xyxy = filtered_gobs['xyxy'][idx]
            view_score = calculate_view_score(mask_area, xyxy, img_width, img_height)
            
            detection = {
                'idx' : idx + 1, # 临时初始 ID，后续会在匹配阶段重分配
                'class_id' : [filtered_gobs['class_id'][idx]],
                'class_label' : detection_class_labels[idx],
                'mask' : filtered_gobs['mask'][idx],
                'mask_area' : filtered_gobs['mask'][idx].sum(),
                'clip_ft' : filtered_gobs['image_feats'][idx],
                'image_crops' : filtered_gobs['image_crops'][idx],
                'num_detections' : 1,
                'best_view' : time_idx,
                'view_score' : view_score,
                'confidence': filtered_gobs['confidence'][idx],
                'xyxy': xyxy
            }
            detections.append(detection)
    
    # 理想情况假设 SAM 分割准确，移除容易造成错误合并的 merge_curr_detection_itself

    return detections


def select_idx_gaussian(params: dict, 
                        select_idx,
                        color_book,
                        if_first_frame):
    
    # TODO：剔除 indice_idx 中 max_probs 小于 0.9 的点

    # Ensure object_idx is a CPU numpy array for comparisons
    if isinstance(params['object_idx'], torch.Tensor):
        obj_idx_arr = params['object_idx'].detach().cpu().numpy()
    else:
        obj_idx_arr = params['object_idx']

    # 除了 indice 都设为黑色 阻挡问题
    indice = np.where(obj_idx_arr == select_idx)[0]
    
    select_params = {
        'means3D': params['means3D'][indice],
        'rgb_colors': params['rgb_colors'][indice],
        'unnorm_rotations': params['unnorm_rotations'][indice],
        'logit_opacities': params['logit_opacities'][indice],
        'log_scales': params['log_scales'][indice],
        'cam_unnorm_rots': params['cam_unnorm_rots'],
        'cam_trans': params['cam_trans'],
    }

    return select_params


def select_idx_occupy_gaussian(params: dict, 
                        select_idx,
                        color_book,
                        if_first_frame):
    
    # TODO：剔除 indice_idx 中 max_probs 小于 0.9 的点

    # Ensure object_idx is a CPU numpy array for comparisons
    if isinstance(params['object_idx'], torch.Tensor):
        obj_idx_arr = params['object_idx'].detach().cpu().numpy()
    else:
        obj_idx_arr = params['object_idx']

    # 除了 indice 都设为黑色 阻挡问题
    indice = np.where(obj_idx_arr == select_idx)[0]
    un_indice = np.where(obj_idx_arr != select_idx)[0]

    # Convert indices to torch tensors on the same device as rgb_colors for safe indexing
    rgb = params['rgb_colors']
    device = rgb.device if isinstance(rgb, torch.Tensor) else None
    if device is not None:
        if indice.size > 0:
            idx_t = torch.as_tensor(indice, dtype=torch.long, device=device)
            params['rgb_colors'][idx_t] = torch.ones_like(params['rgb_colors'][idx_t])
        if un_indice.size > 0:
            un_idx_t = torch.as_tensor(un_indice, dtype=torch.long, device=device)
            params['rgb_colors'][un_idx_t] = torch.zeros_like(params['rgb_colors'][un_idx_t])
    else:
        # rgb_colors is not a tensor (unexpected), fall back to numpy ops
        params['rgb_colors'][indice] = np.ones_like(params['rgb_colors'][indice])
        params['rgb_colors'][un_indice] = np.zeros_like(params['rgb_colors'][un_indice])

    select_params = {
        'means3D': params['means3D'],
        'rgb_colors': params['rgb_colors'],
        'unnorm_rotations': params['unnorm_rotations'],
        'logit_opacities': params['logit_opacities'],
        'log_scales': params['log_scales'],
        'cam_unnorm_rots': params['cam_unnorm_rots'],
        'cam_trans': params['cam_trans'],
    }

    return select_params


def render_curr_frame_with_idx(params: dict, 
                               time_idx, 
                               curr_data: dict, 
                               objects: MapObjectList,
                               color_book,
                               if_first_frame: bool = False):
    """
    渲染当前相机视角下的所有物体，并处理遮挡问题（Z-Buffer / 深度测试）。
    通过一次性渲染所有带有 ID 颜色的高斯点，能够自动利用渲染器的深度排序实现物理遮挡。
    """
    curr_frame_objects = MapObjectList()
    
    if len(objects) == 0:
        return curr_frame_objects

    with torch.no_grad():
        # 1. 修改所有点的颜色为其对应的 ID 颜色 (只对有关联物体的点)
        # 备份原颜色
        orig_colors = params['rgb_colors'].clone() if isinstance(params['rgb_colors'], torch.Tensor) else params['rgb_colors'].copy()
        
        # Ensure object_idx is available as tensor
        if isinstance(params['object_idx'], np.ndarray):
            obj_idx_t = torch.from_numpy(params['object_idx']).cuda().long()
        else:
            obj_idx_t = params['object_idx'].cuda().long()
            
        # 将 rgb_colors 转换为 tensor (如果还不是)
        if isinstance(params['rgb_colors'], np.ndarray):
            params['rgb_colors'] = torch.from_numpy(params['rgb_colors']).cuda().float()
            
        # 背景点或未关联的点设为全黑，不参与 ID 掩码生成
        params['rgb_colors'].fill_(0.0)
        
        # 赋予每个物体专属的唯一颜色（通过 color_book）
        # 这里用颜色作为 ID 的标识
        color_to_id = {}
        for obj in objects:
            idx = obj['idx']
            if isinstance(idx, (list, tuple)):
                idx = idx[0]
            if hasattr(idx, "item"):
                idx = idx.item()
            try:
                idx = int(idx)
            except TypeError:
                idx = int(idx[0])
            
            # 同样更新 obj 里的 idx，防止后续报错
            obj['idx'] = idx

            while idx >= len(color_book):
                # 把生成的颜色转成标准 list 以免触发 tensor 判定问题
                color_book.append(np.random.rand(3).tolist())
            
            # 颜色已经是 0-1 的范围
            color_tensor = torch.tensor(color_book[idx], dtype=torch.float32).cuda()
            
            # 将属于该 idx 的高斯点染成特定颜色
            mask = (obj_idx_t == idx).squeeze()
            if mask.any():
                params['rgb_colors'][mask] = color_tensor
                
            # 记录 color tensor 的 hash (或近似值) 到 ID 的映射，用于反解
            # 因为渲染会有插值和混合，我们后续通过计算欧氏距离来找最近的 ID 颜色
            color_to_id[tuple(np.round(color_tensor.cpu().numpy(), 3))] = idx
            
        # 2. 全局渲染一次 (自动处理了 Z-Buffer 遮挡)
        transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
        rendervar = transformed_params_for_detection(params, transformed_gaussians)
        
        rasterizer = Renderer(raster_settings=curr_data['cam'])
        # rendered_image 形状为 (3, H, W)
        rendered_image, _, _ = rasterizer(**rendervar)
        
        # 恢复原颜色
        params['rgb_colors'] = orig_colors
        
        # 3. 解析渲染结果，提取每个物体的可见 Mask
        # rendered_image 的背景是 0
        img_h, img_w = rendered_image.shape[1], rendered_image.shape[2]
        rendered_image_flat = rendered_image.permute(1, 2, 0).reshape(-1, 3) # (H*W, 3)
        
        # 过滤掉背景像素 (全 0)
        valid_mask = (rendered_image_flat > 0.01).any(dim=1)
        
        # 解析每个有效像素属于哪个物体 (寻找最接近的 ID 颜色)
        # 这里由于高斯渲染的 alpha 混合，边缘可能会有颜色渐变，我们用 L2 距离判定
        for obj in tqdm(objects, desc="Extracting Occlusion-Aware Masks"):
            idx = obj['idx']
            if isinstance(idx, (list, tuple)):
                idx = idx[0]
            if hasattr(idx, "item"):
                idx = idx.item()
            try:
                idx = int(idx)
            except TypeError:
                idx = int(idx[0])
            
            target_color = torch.tensor(color_book[idx], dtype=torch.float32).cuda()
            
            # 计算该物体颜色与所有有效像素的颜色距离
            color_diff = torch.norm(rendered_image_flat[valid_mask] - target_color, dim=1)
            
            # 阈值判定：如果颜色非常接近，认为是该物体的像素
            # 由于 alpha 混合，阈值不能太小
            pixel_belongs_to_obj = (color_diff < 0.1)
            
            # 重建 2D Mask
            full_mask = torch.zeros(img_h * img_w, dtype=torch.bool).cuda()
            valid_indices = torch.where(valid_mask)[0]
            full_mask[valid_indices[pixel_belongs_to_obj]] = True
            full_mask = full_mask.view(img_h, img_w).cpu().numpy()
            
            mask_pixel_count = full_mask.sum().item()
            
            # 只有可见像素超过一定数量，才认为该物体在当前视角下是"可见"的
            if mask_pixel_count > 200:
                # 重新计算 in_frustum (原逻辑保留)
                select_params = select_idx_gaussian(params, idx, color_book, if_first_frame)
                transformed_select_gaussians = transform_to_frame(select_params, time_idx, gaussians_grad=False, camera_grad=False)
                in_frustum = points_in_frustum(transformed_select_gaussians['means3D'], curr_data['intrinsics'], 
                                           curr_data['cam'].image_width, curr_data['cam'].image_height, 0.1, 100)
                
                # 兼容旧格式，生成单物体的 color_mask 用于兼容 check_update 等接口
                # 这里的 color_mask 是带遮挡效果的
                color_mask_tensor = torch.zeros((3, img_h, img_w), dtype=torch.float32).cuda()
                color_mask_tensor[:, full_mask] = target_color.unsqueeze(-1)

                curr_obj_dict = {
                    'idx' : idx,
                    'mask' : full_mask,
                    'mask_area' : mask_pixel_count,
                    'color_mask' : color_mask_tensor,
                    'clip_ft' : obj['clip_ft'],
                    'num_detections' : obj['num_detections'],
                    'in_frustum' : in_frustum
                }
                curr_frame_objects.append(curr_obj_dict)
            else:
                print(f"Object {idx} is completely occluded or outside view (visible pixels: {mask_pixel_count}).")

    return curr_frame_objects


def points_in_frustum(points_world, intrinsics, width, height, near, far):
    """
    判断多个点是否在相机视锥体内
    
    参数:
    points_world : np.array (N, 3) - 世界坐标系中的多个3D点
    width : int - 图像宽度(像素)
    height : int - 图像高度(像素)
    near : float - 近裁剪面距离
    far : float - 远裁剪面距离
    
    返回:
    in_frustum : np.array (N,) bool - 每个点是否在视锥体内
    """
    # 转换为numpy数组
    points_cam = points_world
    
    # 提取坐标分量
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    
    # 深度检查 (向量化)
    depth_ok = (z > 0) & (z >= near) & (z <= far)
    
    # 提取内参
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # 计算投影坐标 (避免除零错误)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
    
    # 图像边界检查 (向量化)
    in_image = (u >= 0) & (u <= width) & (v >= 0) & (v <= height)
    
    # 综合条件：深度有效且在图像范围内
    in_frustum = depth_ok & in_image
    
    # 处理无效点（z<=0）
    in_frustum[torch.isnan(u) | torch.isnan(v)] = False

    return in_frustum


def update_curr_object_visibility(params: dict, 
                                  time_idx, 
                                  curr_data: dict, 
                                  objects: MapObjectList,
                                  color_book,
                                  if_first_frame: bool = False):

    with torch.no_grad():
        for i, map_object in enumerate(objects):

            params_wb = select_idx_occupy_gaussian(params, map_object['idx'], color_book, if_first_frame)
            transformed_wb_gaussians = transform_to_frame(params_wb, time_idx, gaussians_grad=False, camera_grad=False)
            
            select_params = select_idx_gaussian(params, map_object['idx'], color_book, if_first_frame)
            transformed_select_gaussians = transform_to_frame(select_params, time_idx, gaussians_grad=False, camera_grad=False)
            
            rasterizer = Renderer(raster_settings=curr_data['cam'])
            # visible = rasterizer.markVisible(transformed_select_gaussians['means3D'])
            visible = points_in_frustum(transformed_select_gaussians['means3D'], curr_data['intrinsics'], 
                                           curr_data['cam'].image_width, curr_data['cam'].image_height, 0.1, 100)

            select_rendervar = transformed_params_for_detection(params_wb, transformed_wb_gaussians)
            object_mask, _, _, = rasterizer(**select_rendervar)
            bool_mask = (object_mask != 0).any(dim=0).cpu().numpy()
            mask_pixel_count = bool_mask.sum().item()

            visibility = (mask_pixel_count / (curr_data['cam'].image_width * curr_data['cam'].image_height)) * (torch.sum(visible) / transformed_select_gaussians['means3D'].shape[0])
                
            if mask_pixel_count > 200:
                if "visibility" not in objects[i]:
                    objects[i]['visibility'] = visibility
                    objects[i]['best_view'] = time_idx
                else:
                    if visibility > map_object['visibility']:
                        objects[i]['visibility'] = visibility 
                        objects[i]['best_view'] = time_idx
                        objects[i]['mask'] = bool_mask
                        objects[i]['mask_area'] = mask_pixel_count
                        # plt.title("Boolean Mask")
                        # plt.title(f"Bool mask : {map_object['idx']}")  
                        # plt.imshow(bool_mask, cmap='gray')  
                        # plt.axis('off')
                        # plt.show()

    
    return objects

def update_curr_object_visibility_1(params: dict, 
                                  time_idx, 
                                  curr_data: dict, 
                                  objects: MapObjectList,
                                  color_book,
                                  if_first_frame: bool = False):

    with torch.no_grad():
        for i, map_object in enumerate(objects):

            select_params = select_idx_occupy_gaussian(params, map_object['idx'], color_book, if_first_frame)

            transformed_select_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
            select_rendervar = transformed_params_for_detection(params, transformed_select_gaussians)
            rasterizer = Renderer(raster_settings=curr_data['cam'])
            object_mask, _, _, = rasterizer(**select_rendervar)
            visible = rasterizer.markVisible(select_params['means3D'])
            
            bool_mask = (object_mask != 0).any(dim=0).cpu().numpy()
            mask_pixel_count = bool_mask.sum().item()

            visibility = (mask_pixel_count / (curr_data['cam'].image_width * curr_data['cam'].image_height)) * (torch.sum(visible) / transformed_select_gaussians['means3D'].shape[0])
                
            if mask_pixel_count > 200:
                if "visibility" not in objects[i]:
                    objects[i]['visibility'] = visibility
                    objects[i]['best_view'] = time_idx
                else:
                    if visibility > map_object['visibility']:
                        objects[i]['visibility'] = visibility 
                        objects[i]['best_view'] = time_idx
                        objects[i]['mask'] = bool_mask
                        objects[i]['mask_area'] = mask_pixel_count
                    plt.title("Boolean Mask")
                    plt.title(f"Bool mask : {map_object['idx']}")  
                    plt.imshow(bool_mask, cmap='gray')  
                    plt.axis('off')
                    plt.show()

    
    return objects


def compute_mask_iou_similarities(detections: DetectionList, 
                                  curr_cam_mapobjects: MapObjectList,
                                  privilege: bool = False):

    detection_masks = detections.get_stacked_values_torch('mask').unsqueeze(1)
    mapobject_masks = curr_cam_mapobjects.get_stacked_values_torch('mask').unsqueeze(0).cpu()

    intersection = (detection_masks & mapobject_masks).float().sum(dim=(2, 3))
    union = (detection_masks | mapobject_masks).float().sum(dim=(2, 3))
    iou = intersection / union

    if privilege:
        # 计算mapobject占detection的比例：地图物体被detection完全覆盖
        map_sum = mapobject_masks.float().sum(dim=(2, 3))
        object_in_detection = intersection / map_sum
        return iou, object_in_detection, map_sum
    
    return iou


def compute_clip_features_similarities(detections: DetectionList, 
                                       curr_cam_mapobjects: MapObjectList):
    
    detection_features = detections.get_stacked_values_torch('clip_ft').unsqueeze(-1)
    mapobject_features = curr_cam_mapobjects.get_stacked_values_torch('clip_ft').T.unsqueeze(0)

    clip_features_similarities = F.cosine_similarity(detection_features, mapobject_features, dim=1)

    return clip_features_similarities
    

def aggregate_similarities(iou_similarities: torch.Tensor, 
                           features_similarities: torch.Tensor,
                           similarity_bias: float):
    
    similarities = (1 + similarity_bias) * iou_similarities + (1 - similarity_bias) * features_similarities

    return similarities / 2 


def match_detections_to_mapobjects(similarities: torch.Tensor,
                                   curr_cam_mapobjects: MapObjectList,
                                   objects: MapObjectList,
                                   similarity_threshold: float):
    
    match_indices = []
    for detected_obj_idx in range(similarities.shape[0]):
        max_sim_value = similarities[detected_obj_idx].max()
        if max_sim_value <= similarity_threshold:
            match_indices.append(None)
        else:
            i = curr_cam_mapobjects[similarities[detected_obj_idx].argmax().item()]['idx']
            for index, object in enumerate(objects):
                if object['idx'] == i:
                    match_indices.append(index)
                    break

    return match_indices


def merge_obj2_into_obj1(obj1: dict, obj2: dict):

    obj1['class_id'].extend(obj2['class_id'])

    merged_obj = {
        'idx' : obj1['idx'],
        'class_id' : obj1['class_id'],
        'class_label' : obj1['class_label'],
        'mask' : obj1['mask'] | obj2['mask'],
        'mask_area' : (obj1['mask'] | obj2['mask']).sum(),
        'clip_ft' : (obj1['clip_ft'] * obj1['num_detections'] + obj2['clip_ft'] * obj2['num_detections']) / (obj1['num_detections'] + obj2['num_detections']),
        'image_crops' : obj1['image_crops'],
        'num_detections' : obj1['num_detections'] + obj2['num_detections'],
        'best_view' : obj1['best_view']
    }

    return merged_obj

def merge_obj2_into_obj1_for_merge(obj1: dict, obj2: dict):

    # Use view_score if available, otherwise fallback to mask_area
    score1 = obj1.get('view_score', obj1['mask_area'])
    score2 = obj2.get('view_score', obj2['mask_area'])

    if score1 > score2:
        mask_to_keep = obj1['mask']
        mask_area_to_keep = obj1['mask_area']
        best_view_to_keep = obj1['best_view']
        image_crops_to_keep = obj1['image_crops']
        view_score_to_keep = obj1.get('view_score', score1)
    else:
        mask_to_keep = obj2['mask']
        mask_area_to_keep = obj2['mask_area']
        best_view_to_keep = obj2['best_view']
        image_crops_to_keep = obj2['image_crops']
        view_score_to_keep = obj2.get('view_score', score2)

    obj1['class_id'].extend(obj2['class_id'])
    merged_obj = {
        'idx' : obj1['idx'],
        'class_id' : obj1['class_id'],
        'mask' : mask_to_keep,
        'mask_area' : mask_area_to_keep,
        'clip_ft' : (obj1['clip_ft'] * obj1['num_detections'] + obj2['clip_ft'] * obj2['num_detections']) / (obj1['num_detections'] + obj2['num_detections']),
        'image_crops' : image_crops_to_keep,
        'num_detections' : obj1['num_detections'] + obj2['num_detections'],
        'best_view' : best_view_to_keep,
        'view_score' : view_score_to_keep
        # 'visibility' : obj1['visibility'],
        # 'best_view' : obj1['best_view']
    }

    return merged_obj


def merge_obj_matches(detections: DetectionList, 
                      objects: MapObjectList,
                      curr_cam_mapobjects: MapObjectList,  
                      match_indices: list,
                      privilege_similarities: torch.Tensor,
                      currmap_sum: torch.Tensor,
                      time_idx):
    new_objects_idx = []
    privilege_object = []
    for detected_obj_idx, existing_obj_match_idx in enumerate(match_indices):
        if existing_obj_match_idx is None:
            if privilege_similarities[detected_obj_idx].max() > 0.7:
                # 虽然没匹配上地图中的物体，但是覆盖了地图中的某个物体，将该物体对应到地图物体的idx
                idx = curr_cam_mapobjects[privilege_similarities[detected_obj_idx].argmax().item()]['idx']
                for index, object in enumerate(objects):
                    if object['idx'] == idx:
                        merged_obj = merge_obj2_into_obj1_for_merge(
                            obj1=objects[index],
                            obj2=detections[detected_obj_idx],
                        )
                        objects[index] = merged_obj
                        match_indices[detected_obj_idx] = index
                        # with open('/home/coastz/codes/SplaTAM/experiments/Replica/room1_0/new_objects_idx.txt', 'a') as f:
                        #     f.write(f" 覆盖了：{idx}\n")  
                        break
            else:
                # track the new object detection:TODO:增加一个记录当前最大idx的变量
                detections[detected_obj_idx]['idx'] = objects[-1]['idx'] + 1
                objects.append(detections[detected_obj_idx])
                match_indices[detected_obj_idx] = len(objects) - 1
                new_objects_idx.append(detections[detected_obj_idx]['idx'])
        else:
            detected_obj = detections[detected_obj_idx]
            matched_obj = objects[existing_obj_match_idx]
            merged_obj = merge_obj2_into_obj1_for_merge(
                obj1=matched_obj,
                obj2=detected_obj,
            )
            objects[existing_obj_match_idx] = merged_obj

            # 如果当前匹配到了地图中的物体，但是覆盖了地图中的其他物体
            # 赋予被匹配到的地图物体特权
            if privilege_similarities[detected_obj_idx].max() > 0.98 and currmap_sum[0][privilege_similarities[detected_obj_idx].argmax().item()] > 1000:
            # if privilege_similarities[detected_obj_idx].max() > 0.98 :
                idx = curr_cam_mapobjects[privilege_similarities[detected_obj_idx].argmax().item()]['idx']
                for index, object in enumerate(objects):
                    if object['idx'] == idx and (index != existing_obj_match_idx):
                        # tuple: (privilege_obj_idx, new_obj_idx) clip_ft
                        clip_sim = F.cosine_similarity(torch.from_numpy(objects[existing_obj_match_idx]['clip_ft']), 
                                                       torch.from_numpy(objects[index]['clip_ft']), dim=0)
                        if clip_sim > 0.3:
                            privilege_object.append((objects[existing_obj_match_idx]['idx'], idx))
                            
                            img_crop = np.array(detections[detected_obj_idx]['image_crops'])
                            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
                            # cv2.imwrite(f"/home/coastz/codes/SplaTAM/experiments/Replica/room1_0/{time_idx}_{detected_obj_idx}.jpg", img_crop)

                            # 将mask保存为图片
                            mask = curr_cam_mapobjects[privilege_similarities[detected_obj_idx].argmax().item()]['mask']  # np (680, 1200) bool
                            mask = mask.astype(np.uint8) * 255
                            # cv2.imwrite(f"/home/coastz/codes/SplaTAM/experiments/Replica/room1_0/{time_idx}_{privilege_similarities[detected_obj_idx].argmax().item()}_mask.jpg", mask)
                            img_crop = np.array(objects[existing_obj_match_idx]['image_crops'])
                            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
                            # cv2.imwrite(f"/home/coastz/codes/SplaTAM/experiments/Replica/room1_0/{privilege_object[0][0]}.jpg", img_crop)
                            img_crop = np.array(objects[index]['image_crops'])
                            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
                            # cv2.imwrite(f"/home/coastz/codes/SplaTAM/experiments/Replica/room1_0/{privilege_object[0][1]}.jpg", img_crop)

                            # with open('/home/coastz/codes/SplaTAM/experiments/Replica/room1_0/new_objects_idx.txt', 'a') as f:
                            #     f.write(f"合并了：{privilege_object} {clip_sim}\n")   
                        break
                

    return objects, match_indices, new_objects_idx, privilege_object


def compute_similarities_and_merge(detections: DetectionList, 
                                   curr_cam_mapobjects: MapObjectList,
                                   objects: MapObjectList,
                                   color_book,
                                   cfg: dict,
                                   time_idx,
                                   H, W):
    # TODO: 当前帧检测到mask太小的目标，不参与合并，避免影响匹配！！！
    # 计算detections与当前帧的mapobjects之间的相似度
    '''
        TODO: 是否增加3d重合度的计算？ 后期再考虑
              添加计算mapobject占detection的比例：
                    地图物体被detection完全覆盖，并且有一定相似度，将地图中的物体合并到detection中
                    privilege detection 允许更新被占用的点
    '''
    if len(detections) == 0:
        curr_objects_idx = []
        new_objects_idx = []
        curr_idx_mask = np.zeros((H, W), dtype=np.int8) 
        curr_features_mask = torch.zeros((H, W, 3), dtype=torch.float32)

        return curr_idx_mask, curr_features_mask, objects, curr_objects_idx, new_objects_idx


    if len(curr_cam_mapobjects) != 0 :
        iou_similarities, privilege_similarities, currmap_sum = compute_mask_iou_similarities(detections, curr_cam_mapobjects, True)
        features_similarities = compute_clip_features_similarities(detections, curr_cam_mapobjects)
        similarities = aggregate_similarities(iou_similarities, features_similarities, cfg['similarity_bias'])
        # 得到detecions与objects的匹配关系
        match_indices = match_detections_to_mapobjects(similarities, curr_cam_mapobjects, objects, cfg['similarity_threshold'])
        # 将当前帧合并到地图中
        objects, match_indices, new_objects_idx, privilege_object = merge_obj_matches(detections, objects, curr_cam_mapobjects, 
                                                                                      match_indices, privilege_similarities, currmap_sum, time_idx)
    else:
        match_indices = []
        new_objects_idx = []
        privilege_object = []
        for detected_obj_idx, _ in enumerate(detections):
            detections[detected_obj_idx]['idx'] = objects[-1]['idx'] + 1
            objects.append(detections[detected_obj_idx])
            match_indices.append(len(objects) - 1)
            new_objects_idx.append(detections[detected_obj_idx]['idx'])


    curr_idx_mask = np.zeros((detections[0]['mask'].shape[0], detections[0]['mask'].shape[1]), dtype=np.int8) 
    curr_features_mask = torch.zeros((detections[0]['mask'].shape[0], detections[0]['mask'].shape[1], 3), dtype=torch.float32)
    curr_objects_idx = []
    for detected_obj_idx, existing_obj_match_idx in enumerate(match_indices):
        mask = detections[detected_obj_idx]['mask'] > 0
        curr_idx_mask[mask] = objects[existing_obj_match_idx]['idx']
        curr_features_mask[mask] = torch.tensor(color_book[objects[existing_obj_match_idx]['idx']], dtype=torch.float32)
        curr_objects_idx.append(objects[existing_obj_match_idx]['idx'])
        # # debug
        # plt.subplot(1, 2, 1) 
        # plt.title("Boolean Mask")
        # plt.imshow(detections[detected_obj_idx]['mask'], cmap='gray')  
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.title(f"curr_cam_mapobjects Mask")
        # for obj in curr_cam_mapobjects:
        #     if obj['idx'] == objects[existing_obj_match_idx]['idx']:
        #         plt.imshow(obj['mask'], cmap='gray')  
        #         break
        # plt.axis('off')
        
        # plt.tight_layout()  # 自动调整子图间距
        # plt.show()

    return curr_idx_mask, curr_features_mask, objects, curr_objects_idx, new_objects_idx

def match_and_assign_id(
    detections: list, 
    curr_cam_mapobjects: MapObjectList, 
    objects: MapObjectList, 
    curr_data: dict,
    cfg: dict,
    color_book: list,
    time_idx: int,
    iou_threshold=0.5
):
    """
    步骤2：将当前帧检测出的 2D Mask 与 3D 场景投影下来的 Mask 进行匹配并分配 ID。
    融合 Mask IoU、CLIP 特征相似度、以及 3D 重合度进行综合判定。
    返回: curr_idx_mask, curr_features_mask, objects, curr_objects_idx, new_objects_idx
    """
    H, W = curr_data['im'].shape[1], curr_data['im'].shape[2]
    curr_objects_idx = []
    new_objects_idx = []
    
    if len(detections) == 0:
        curr_idx_mask = np.zeros((H, W), dtype=np.int8)
        curr_features_mask = torch.zeros((H, W, 3), dtype=torch.float32)
        return curr_idx_mask, curr_features_mask, objects, curr_objects_idx, new_objects_idx

    if len(curr_cam_mapobjects) == 0:
        # 场景图为空，直接全部赋新 ID
        for det in detections:
            new_id = objects[-1]['idx'] + 1 if len(objects) > 0 else 1
            det['idx'] = new_id
            det['best_view'] = time_idx # 初始化 best_view
            objects.append(det)
            new_objects_idx.append(new_id)
            curr_objects_idx.append(new_id)
            
        # 构建 Mask 和 Features
        curr_idx_mask = np.zeros((H, W), dtype=np.int8)
        curr_features_mask = torch.zeros((H, W, 3), dtype=torch.float32)
        for det in detections:
            mask = det['mask'] > 0
            curr_idx_mask[mask] = det['idx']
            while det['idx'] >= len(color_book):
                color_book.append(np.random.rand(3).tolist())
            curr_features_mask[mask] = torch.tensor(color_book[det['idx']], dtype=torch.float32)
            
        return curr_idx_mask, curr_features_mask, objects, curr_objects_idx, new_objects_idx

    # 1. 计算 Mask IoU 相似度 (N, M)
    det_masks = torch.from_numpy(np.array([d['mask'] for d in detections])).cuda()
    map_masks = torch.from_numpy(np.array([o['mask'] for o in curr_cam_mapobjects])).cuda()
    
    intersection = (det_masks.unsqueeze(1) & map_masks.unsqueeze(0)).float().sum(dim=(2, 3))
    union = (det_masks.unsqueeze(1) | map_masks.unsqueeze(0)).float().sum(dim=(2, 3))
    iou_matrix = intersection / (union + 1e-6)
    
    # 2. 计算 CLIP 特征相似度 (N, M)
    det_features = torch.from_numpy(np.array([d['clip_ft'] for d in detections])).view(len(detections), -1).cuda()
    map_features = torch.from_numpy(np.array([o['clip_ft'] for o in curr_cam_mapobjects])).view(len(curr_cam_mapobjects), -1).cuda()
    
    det_features_norm = F.normalize(det_features, dim=1)
    map_features_norm = F.normalize(map_features, dim=1)
    clip_matrix = torch.mm(det_features_norm, map_features_norm.T)
    
    # 3. 计算 3D 空间相似度 (N, M)
    depth_map = curr_data['depth'][0].cuda() # (H, W)
    intrinsics = curr_data['intrinsics']
    fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]
    
    y_grid, x_grid = torch.meshgrid(torch.arange(H).cuda(), torch.arange(W).cuda(), indexing='ij')
    
    z_cam = depth_map
    x_cam = (x_grid - cx) * z_cam / fx
    y_cam = (y_grid - cy) * z_cam / fy
    pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1) # (H, W, 3)
    
    spatial_sim_matrix = torch.zeros((len(detections), len(curr_cam_mapobjects))).cuda()
    
    for i, det_mask in enumerate(det_masks):
        det_pts = pts_cam[det_mask & (z_cam > 0)]
        if len(det_pts) == 0:
            continue
        det_center = det_pts.mean(dim=0)
        
        for j, map_mask in enumerate(map_masks):
            map_pts = pts_cam[map_mask & (z_cam > 0)]
            if len(map_pts) == 0:
                continue
            map_center = map_pts.mean(dim=0)
            
            # 计算欧氏距离
            dist = torch.norm(det_center - map_center)
            # 距离转换为相似度 (使用高斯衰减)
            spatial_sim_matrix[i, j] = torch.exp(-2.0 * dist)
            
    # 综合相似度
    iou_weight = cfg.get('iou_weight', 0.4)
    clip_weight = cfg.get('clip_weight', 0.4)
    spatial_weight = cfg.get('spatial_weight', 0.2)
    
    total_sim_matrix = iou_weight * iou_matrix + clip_weight * clip_matrix + spatial_weight * spatial_sim_matrix
    
    # 4. 贪心匹配
    total_sim_np = total_sim_matrix.cpu().numpy()
    matched_det_indices = set()
    matched_map_indices = set()
    
    flat_indices = np.argsort(total_sim_np, axis=None)[::-1]
    
    sim_thresh = cfg.get('similarity_threshold', iou_threshold)
    
    for flat_idx in flat_indices:
        det_idx, map_idx = np.unravel_index(flat_idx, total_sim_np.shape)
        sim_val = total_sim_np[det_idx, map_idx]
        
        if sim_val < sim_thresh:
            break 
            
        if det_idx not in matched_det_indices and map_idx not in matched_map_indices:
            matched_obj_id = curr_cam_mapobjects[map_idx]['idx']
            if hasattr(matched_obj_id, "item"):
                matched_obj_id = matched_obj_id.item()
            try:
                matched_obj_id = int(matched_obj_id)
            except TypeError:
                matched_obj_id = int(matched_obj_id[0])
                
            detections[det_idx]['idx'] = matched_obj_id
            curr_objects_idx.append(matched_obj_id)
            
            for obj in objects:
                if obj['idx'] == matched_obj_id:
                    # 使用 merge_obj2_into_obj1_for_merge 逻辑，保留 best_view 下的属性
                    merged_obj = merge_obj2_into_obj1_for_merge(obj, detections[det_idx])
                    
                    # 更新 obj 的所有属性
                    for k, v in merged_obj.items():
                        obj[k] = v
                    
                    break
            
            matched_det_indices.add(det_idx)
            matched_map_indices.add(map_idx)
            
    # 5. 处理未匹配上的 detections (赋予新 ID)
    for det_idx, det in enumerate(detections):
        if det_idx not in matched_det_indices:
            new_id = objects[-1]['idx'] + 1 if len(objects) > 0 else 1
            if hasattr(new_id, "item"):
                new_id = new_id.item()
            try:
                new_id = int(new_id)
            except TypeError:
                new_id = int(new_id[0])
                
            det['idx'] = new_id
            det['best_view'] = time_idx # 初始化新物体的 best_view
            objects.append(det)
            new_objects_idx.append(new_id)
            curr_objects_idx.append(new_id)

    # 6. 构建当前帧的 idx_mask 和 features_mask
    curr_idx_mask = np.zeros((H, W), dtype=np.int8)
    curr_features_mask = torch.zeros((H, W, 3), dtype=torch.float32)
    
    for det in detections:
        mask = det['mask'] > 0
        idx = det['idx']
        if hasattr(idx, "item"):
            idx = idx.item()
        try:
            idx = int(idx)
        except TypeError:
            idx = int(idx[0])
            
        curr_idx_mask[mask] = idx
        
        while idx >= len(color_book):
            color_book.append(np.random.rand(3).tolist())
        curr_features_mask[mask] = torch.tensor(color_book[idx], dtype=torch.float32)

    return curr_idx_mask, curr_features_mask, objects, curr_objects_idx, new_objects_idx


def check_update(color, depth, curr_cam_mapobjects, detections=None, cfg=None):

    objects_to_remove = []
    for i, curr_obj in enumerate(curr_cam_mapobjects):
        mask = torch.from_numpy(curr_obj['mask']).cuda()
        weighted_im = curr_obj['color_mask'][:,mask].cpu().numpy()  
        weighted_gt_im = color[:,mask].cpu().numpy()
        
        ssim_scores = 0
        for c in range(3):  # 遍历R、G、B通道
            gt_channel_masked = weighted_gt_im[c, ...]
            pred_channel_masked = weighted_im[c, ...] 
            # 计算SSIM（需将数据范围归一化到0-1或0-255）
            score = ssim(
                gt_channel_masked, pred_channel_masked,
                data_range=1,  # 如果图像是uint8类型，范围为0-255
                channel_axis=None  # 单通道无需指定
            )
            ssim_scores += score

        ssim_scores = ssim_scores / 3
        if ssim_scores < 0.15:
            objects_to_remove.append(curr_obj['idx'])

        vis1 =  curr_obj['color_mask'] * mask
        vis2 = color * mask
        plt.subplot(1, 2, 1) 
        plt.title("dynamic changes")
        plt.imshow(vis1.permute(1, 2, 0).detach().cpu().numpy())
        plt.axis('off')

        plt.subplot(1, 2, 2) 
        plt.title("dynamic changes gt")
        plt.imshow(vis2.permute(1, 2, 0).detach().cpu().numpy())
        plt.axis('off')
        plt.tight_layout()  # 自动调整子图间距
        plt.show()

    # iou_similarities = compute_mask_iou_similarities(curr_cam_mapobjects, detections, False)
    # features_similarities = compute_clip_features_similarities(curr_cam_mapobjects, detections)
    # similarities = aggregate_similarities(iou_similarities, features_similarities, cfg['similarity_bias'])
    # objects_to_remove = []
    # for i, curr_obj in enumerate(curr_cam_mapobjects):
    #     if similarities[i].max() < cfg['similarity_threshold']:
    #         objects_to_remove.append(curr_obj['idx'])
    
    return objects_to_remove



def prune_visible_gaussians(params: dict, variables: dict, curr_data: dict, time_idx: int):
    """
    Prune explicit ghost Gaussians and handle ID conflicts in the dead zone.
    This ensures clean geometry and feature maintenance when mapping_iters=0.
    """
    depth_map = curr_data['depth'][0].cuda() # (H, W)
    H, W = depth_map.shape
    
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    pts_cam = transformed_gaussians['means3D'] # Already in Camera Space
    z_all = pts_cam[:, 2]
    
    intrinsics = curr_data['intrinsics']
    fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]
    
    valid_z = z_all > 0.001
    
    u_all = (pts_cam[:, 0] / z_all * fx + cx).long()
    v_all = (pts_cam[:, 1] / z_all * fy + cy).long()
    
    valid_proj = valid_z & (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H)
    valid_indices = torch.where(valid_proj)[0]
    
    if valid_indices.shape[0] == 0:
        return params, variables
        
    u_valid = u_all[valid_indices]
    v_valid = v_all[valid_indices]
    z_valid = z_all[valid_indices]
    
    d_gt = depth_map[v_valid, u_valid]
    
    # 【核心新增】：过滤掉无效深度（d_gt <= 0）和极端深度（d_gt > 15.0）
    # 室内场景深度一般不超过10米，超过15米通常为无效值或噪声
    valid_depth_mask = (d_gt > 0.001) & (d_gt < 15.0)
    
    # 仅保留拥有有效深度值的点参与后续的差异计算和修剪逻辑
    valid_indices = valid_indices[valid_depth_mask]
    z_valid = z_valid[valid_depth_mask]
    u_valid = u_valid[valid_depth_mask]
    v_valid = v_valid[valid_depth_mask]
    d_gt = d_gt[valid_depth_mask]
    
    diff = z_valid - d_gt
    margin = 0.1
    
    # 1. 情况C: 悬浮幽灵点 (旧点比真实表面近很多) -> 删除
    condition_c = diff < -margin
    
    # 2. 情况A: 死区 (深度基本一致) -> 检查 ID 冲突
    condition_a = torch.abs(diff) <= margin
    
    if isinstance(curr_data['idx_mask'], np.ndarray):
        new_idx_map = torch.from_numpy(curr_data['idx_mask']).cuda().long()
    else:
        new_idx_map = curr_data['idx_mask'].cuda().long()
        
    matched_new_ids = new_idx_map[v_valid, u_valid]
    
    valid_indices_cpu = valid_indices.cpu().numpy()
    current_ids = params['object_idx'][valid_indices_cpu]
    if current_ids.ndim > 1:
        current_ids = current_ids.flatten()
        
    current_ids_tensor = torch.from_numpy(current_ids).cuda().long()
    
    id_conflict_mask = condition_a & (current_ids_tensor != matched_new_ids)
    
    # delete_mask = condition_c | id_conflict_mask
    delete_mask = condition_c
    indices_to_delete = valid_indices[delete_mask].cpu().numpy()
    
    if indices_to_delete.shape[0] > 0:
        print(f"Pruning {indices_to_delete.shape[0]} Gaussians (Ghost/ID Conflict in Dead Zone)")
        keep_mask = np.ones(params['means3D'].shape[0], dtype=bool)
        keep_mask[indices_to_delete] = False
        keep_tensor_mask = torch.from_numpy(keep_mask).cuda()
        
        for k, v in params.items():
            if k in ['cam_unnorm_rots', 'cam_trans']:
                continue
            if isinstance(v, torch.Tensor):
                if v.ndim > 0 and v.shape[0] == keep_tensor_mask.shape[0]:
                    params[k] = torch.nn.Parameter(v[keep_tensor_mask].requires_grad_(v.requires_grad))
            elif isinstance(v, np.ndarray):
                if v.ndim > 0 and v.shape[0] == keep_mask.shape[0]:
                    params[k] = v[keep_mask]
                    
        for k in variables.keys():
            if isinstance(variables[k], torch.Tensor) and variables[k].ndim > 0 and variables[k].shape[0] == keep_tensor_mask.shape[0]:
                variables[k] = variables[k][keep_tensor_mask]

    return params, variables

def degrade_orphan_points(params: dict, objects: MapObjectList):
    """
    Degrade points whose ID no longer exists in the active object list to background (0).
    """
    active_ids = {obj['idx'] for obj in objects}
    active_ids.add(0)  # Always keep background ID valid
    
    current_ids = params['object_idx']
    if current_ids.ndim > 1:
        current_ids = current_ids.flatten()
        
    # Find IDs not in active list
    unique_ids = np.unique(current_ids)
    orphaned_ids = [uid for uid in unique_ids if uid not in active_ids]
    
    num_degraded = 0
    for oid in orphaned_ids:
        orphan_mask = current_ids == oid
        num_degraded += np.sum(orphan_mask)
        current_ids[orphan_mask] = 0
        
    if num_degraded > 0:
        print(f"Degraded {num_degraded} orphan points to background (ID 0)")
        
    # Reshape back if it was reshaped
    if params['object_idx'].ndim > 1:
        params['object_idx'] = current_ids.reshape(params['object_idx'].shape)
    else:
        params['object_idx'] = current_ids
        
    return params




def update_curr_objects_gaussians(params : dict, objects: MapObjectList,
                                  curr_objects_idx: list, new_objects_idx: list, cfg: dict, time_idx: int, curr_data: dict=None):

    # 方案一：Projection-based Association
    # Complexity: O(N) instead of O(N*M)
    
    if curr_data is None:
        # Should not happen if caller is updated
        print("Error: curr_data is None in update_curr_objects_gaussians")
        return params, []

    # Prepare Depth Map for visibility check
    depth_map = curr_data['depth'][0].cuda() # (H, W)
    H, W = depth_map.shape
    
    invaild_new_objects_idx = []
    
    # Project ALL Scene Gaussians
    # transformed_gaussians already computed above (means3D is already in Camera Space)
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    pts_cam = transformed_gaussians['means3D'] # Already in Camera Space
    z_all = pts_cam[:, 2]
    
    intrinsics = curr_data['intrinsics']
    fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]
    
    # Perspective Projection
    # u = fx * x / z + cx
    # v = fy * y / z + cy
    
    # Filter valid projections
    valid_z = z_all > 0.001
    
    u_all = (pts_cam[:, 0] / z_all * fx + cx).long()
    v_all = (pts_cam[:, 1] / z_all * fy + cy).long()
    
    valid_proj = valid_z & (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H)
    valid_indices = torch.where(valid_proj)[0]
    
    if valid_indices.shape[0] == 0:
        return params, new_objects_idx # All new objects invalid if no projection?

    u_valid = u_all[valid_indices]
    v_valid = v_all[valid_indices]
    z_valid = z_all[valid_indices]
    
    # Get New Object ID at these locations
    if isinstance(curr_data['idx_mask'], np.ndarray):
        new_idx_map = torch.from_numpy(curr_data['idx_mask']).cuda().long()
    else:
        new_idx_map = curr_data['idx_mask'].cuda().long()
        
    matched_new_ids = new_idx_map[v_valid, u_valid]
    
    # Check Visibility (Depth Test)
    depth_val = depth_map[v_valid, u_valid]
    # 10cm threshold
    is_visible = (depth_val > 0) & (torch.abs(z_valid - depth_val) < 0.1)
    
    # For each object, we update the gaussians that match
    for object_idx in curr_objects_idx:
        
        # Find Gaussians that project to this object's mask and are visible
        # matched_new_ids is aligned with valid_indices
        obj_mask = (matched_new_ids == object_idx) & is_visible
        
        # These are the indices into `params` that should be updated
        indices_to_update = valid_indices[obj_mask]
        
        if indices_to_update.shape[0] == 0:
            if object_idx in new_objects_idx:
                invaild_new_objects_idx.append(object_idx)
            continue
            
        # Convert to CPU numpy for indexing into params['object_idx'] (which is numpy)
        indices_to_update_cpu = indices_to_update.cpu().numpy()
        
        current_ids = params['object_idx'][indices_to_update_cpu]
        if current_ids.ndim > 1:
            current_ids = current_ids.flatten()
            
        valid_overwrite = (current_ids == 0) | (current_ids == object_idx)
        
        num_candidates = indices_to_update_cpu.shape[0]
        num_valid = np.sum(valid_overwrite)
        
        ratio = num_valid / num_candidates if num_candidates > 0 else 0
        
        if object_idx in new_objects_idx:
            if (num_candidates > cfg['update_gs_num_threshold']) and (ratio > cfg['update_gs_ratio_threshold']):
                final_indices = indices_to_update_cpu[valid_overwrite]
                params['object_idx'][final_indices] = object_idx
                print(f"update new object {object_idx} gs num:{num_candidates}")
            else:
                invaild_new_objects_idx.append(object_idx)
        else:
            # 如果是已有物体，且当前候选点数量极大（例如超过某个阈值），则允许强制覆盖（即忽略 valid_overwrite）
            if num_candidates > cfg.get('force_overwrite_gs_num_threshold', 1000):
                final_indices = indices_to_update_cpu # 强制覆盖
                params['object_idx'][final_indices] = object_idx
                print(f"force updated existing object {object_idx} gs num: {final_indices.shape[0]}")
            else:
                final_indices = indices_to_update_cpu[valid_overwrite]
                params['object_idx'][final_indices] = object_idx
                print(f"updated existing object {object_idx} gs num: {final_indices.shape[0]}")
    
    return params, invaild_new_objects_idx


def slice_invaild_new_objects(invaild_new_objects_idx, objects: MapObjectList, curr_objects_idx, curr_data: dict):
    
    if len(invaild_new_objects_idx) == 0:
        return objects, curr_objects_idx, curr_data
    else:
        for invaild_idx in invaild_new_objects_idx:
            invaild_pix =  curr_data['idx_mask'] == invaild_idx
            curr_data['idx_mask'][invaild_pix] = 0
            curr_data['features'][invaild_pix] = 0

        objects = [obj for obj in objects if obj['idx'] not in invaild_new_objects_idx]
        curr_obj_idx = [cidx for cidx in curr_objects_idx if cidx not in invaild_new_objects_idx]
        
    return  MapObjectList(objects), curr_obj_idx, curr_data


def update_objects_idx(params, objects: MapObjectList, curr_objects_idx, all=False):

    if all:
        for obj in objects:
            obj['all_idx'] = np.where(params['object_idx'] == obj['idx'])[0]
    else:
        for obj in objects:
            if obj['idx'] not in curr_objects_idx:
                continue
            else:
                obj['all_idx'] = np.where(params['object_idx'] == obj['idx'])[0]    

    return objects

def get_curr_objects_pcd(depth, idx_mask, curr_objects_idx, intrinsics, w2c, transform_pts=True, mask=None):

    width, height = depth.shape[2], depth.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
        # pts = pts_cam @ w2c[:3, :3].T + w2c[:3, 3]
    else:
        pts = pts_cam

    point_cld = torch.cat((pts, idx_mask.reshape(-1, 1).to('cuda')), -1)

    if mask is not None:
        point_cld = point_cld[mask]

    curr_objects_pcd = []
    for idx in curr_objects_idx:
        mask = point_cld[:, 3] == idx
        pcd = {
            'idx': idx,
            'points': point_cld[mask][:, :3], # Keep on GPU for faster processing
        }   
        curr_objects_pcd.append(pcd)

    return curr_objects_pcd



def save_keyframe_list(keyframe_list: list, output_dir):
    
    keyframe_list_save_path = Path(output_dir) / "keyframelist.pkl.gz"
    keyframe_list_save_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(keyframe_list_save_path, "wb") as f:
        pickle.dump(keyframe_list, f)

    print(f"Saving keyframe list to: {keyframe_list_save_path}")


def save_objects(params, objects: MapObjectList, dataset, ai_client, lf_config, output_dir):

    prompt_modes = {
            "focal_prompt": "full+focal_crop",
        }
    dam_model = DescribeAnythingModel(
        model_path=lf_config['dam_model_path'],
        conv_mode=lf_config['dam_conv_mode'],
        prompt_mode=prompt_modes.get(lf_config['dam_prompt_mode'], lf_config['dam_prompt_mode']),
    ).to("cuda:0")

    with open(lf_config['obj_prompt_file'], "r") as f:
        obj_category_prompt = f.read()
        
    with open(lf_config['obj_caption_file'], "r") as f:
        obj_caption_prompt = f.read()
        
    objects_img_crop_save_path = f"{output_dir}/objects_img_crop"

    if not os.path.exists(objects_img_crop_save_path):
        os.makedirs(objects_img_crop_save_path)

    # image_crops PIL to numpy 
    caption_generation_time_avg = 0
    caption_generation_count = 0
    dam_time_avg = 0
    dam_time_count = 0
    for i, obj in enumerate(objects):
        caption_start_time = time.time()
        img_crop = np.array(obj['image_crops'])
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{objects_img_crop_save_path}/{obj['idx']}.jpg", img_crop)

        image, _ ,_ ,_ = dataset[obj['best_view']]
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., obj['best_view']].detach())
        curr_cam_tran = params['cam_trans'][..., obj['best_view']].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran

        dam_time_start = time.time()
        outputs = dam_model.get_description(
                Image.fromarray(image.cpu().numpy().astype(np.uint8)),
                obj['mask'],
                "<image>\nDescribe the masked region in detail.",
                temperature=0.2,
                top_p=0.5,
                num_beams=1,
                max_new_tokens=512,
            )
        dam_time_end = time.time()
        dam_time_avg += (dam_time_end - dam_time_start)
        dam_time_count += 1
        if dam_time_count > 1:
            dam_time_avg /= 2
        
        objects[i]['description'] = outputs
        objects[i]['best_view_w2c'] = curr_w2c

        obj_category_parse_prompt = f"Query description: {outputs}"
        messages = [
            {"role": "system", "content": obj_category_prompt},
            {"role": "user", "content": [{"type": "text", "text": obj_category_parse_prompt}]},
        ]
        chat_response = ai_client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct", messages=messages
        )
        answer = chat_response.choices[0].message.content
        answer = answer.replace("'", '"')
        answer = json.loads(answer)
        obj['category'] = answer['Object category']

        obj_caption_parse_prompt = f"Query caption: {obj['category']}"
        messages = [
            {"role": "system", "content": obj_caption_prompt},
            {"role": "user", "content": [{"type": "text", "text": obj_caption_parse_prompt}]},
        ]
        chat_response = ai_client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct", messages=messages
        )
        answer = chat_response.choices[0].message.content
        answer = answer.replace("'", '"')
        answer = json.loads(answer)
        obj['caption'] = answer['Object caption']

        caption_end_time = time.time()
        caption_generation_time_avg += (caption_end_time - caption_start_time)
        caption_generation_count += 1
        if caption_generation_count > 1:
            caption_generation_time_avg /= 2
    
    print(f"Avg Time - DAM: {dam_time_avg:.4f}s")
    print(f"Avg Time - Caption Generation (DAM+Qwen): {caption_generation_time_avg:.4f}s")
    
    objects = objects.to_serializable()
    objects_save_path = Path(output_dir) / "objects.pkl.gz"
    objects_save_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(objects_save_path, "wb") as f:
        pickle.dump(objects, f)

    print(f"Saving map objects to: {objects_save_path}")

def save_relations(params, objects: MapObjectList, dataset, ai_client, lf_config, output_dir):

    # with open(lf_config['rel_prompt_file'], "r") as f:
    #     relation_prompt = f.read()

    relations = []
    dam_time_avg = 0
    dam_count = 0
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects[i + 1:], start=i + 1):
            rel = {}
            rel['obj1_id'] = obj1['idx']
            rel['obj2_id'] = obj2['idx']
            # 获取两物体之间的空间关系

            # 获取两物体之间的潜在关系
            dam_start_time = time.time()
            relation_query = (
                f"You see two masked objects.\n"
                f"Object 1 is: {obj1['category']} ({obj1['description']}).\n"
                f"Object 2 is: {obj2['category']} ({obj2['description']}).\n"
                "Only describe the potential functional or usage relationship "
                "between object 1 and object 2.\n"
                "Focus on how object 1 can be used with or acts on object 2.\n"
                "Do NOT describe their appearance, color, material, or shape.\n"
                "For example, if object 1 is a knife and object 2 is an apple, say that the knife can be used to cut the apple.\n "
                "If there is no clear functional relationship, answer exactly: "
                "'no clear functional relationship'.\n"
                "Answer in ONE short English sentence within 20 words."
            )

            messages = [
                {"role": "system", "content": "You are an intelligent robot assistant specializing in object relation analysis."},
                {"role": "user", "content": relation_query}
            ]
            chat_response = ai_client.chat.completions.create(
                model="qwen2.5-vl-72b-instruct", messages=messages
            )
            outputs = chat_response.choices[0].message.content
            # Add a small delay to prevent immediate rate limit hit
            time.sleep(0.5)
            rel['pretential_relation'] = outputs
            relations.append(rel)
            dam_end_time = time.time()
            dam_time_avg += (dam_end_time - dam_start_time)
            dam_count += 1
            print(f"Relation {obj1['category']}: ({obj1['idx']}) - {obj2['category']}:({obj2['idx']}): {outputs}")
            print(f"Time: {dam_end_time - dam_start_time:.4f}s")
    # 保存关系
    relations_save_path = Path(output_dir) / "relations.pkl.gz"
    relations_save_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(relations_save_path, "wb") as f:
        pickle.dump(relations, f)

    print(f"Avg Time - DAM: {dam_time_avg:.4f}s")
    print(f"Saving map relations to: {relations_save_path}")


def save_objects_relations(params, objects: MapObjectList, dataset, lf_config, output_dir):
    """
    Unified function to generate object descriptions and relations using local LLaVA.
    Replaces save_objects and save_relations.
    """
    total_time_start = time.time()
    
    # Initialize Local VLM
    # Use config or defaults
    model_id = lf_config.get('local_vlm_model', "llava-hf/llava-v1.6-mistral-7b-hf")
    load_4bit = lf_config.get('local_vlm_4bit', False)
    
    # vlm = LocalVLM(model_id=model_id, load_in_4bit=load_4bit)

    objects_img_crop_save_path = Path(output_dir) / "objects_img_crop"
    objects_img_crop_save_path.mkdir(parents=True, exist_ok=True)

    print("=== Generating Object Descriptions and Relations with Local LLaVA ===")
    
    # --- Phase 1: Object Descriptions ---
    object_time_start = time.time()
    for i, obj in enumerate(tqdm(objects, desc="Processing Objects")):
        llava_time_start = time.time()
        best_view_idx = obj['best_view']
        img_crop = np.array(obj['image_crops'])
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{objects_img_crop_save_path}/{obj['idx']}.jpg", img_crop)

        # Prepare for VLM
        image_pil = Image.fromarray(img_crop)
        
        # Generate Description and Category
        prompt = (
            "Describe this object in detail. Then provide a short Category name for it. "
            "The object is typically found in indoor scenes.\n"
            "Briefly describe the object within ten words. Keep the description concise.\n"
            "Focus on the object's appearance, geometry, and material. Do not describe the background or unrelated details.\n"
            "Ensure the description is specific and avoids vague terms.\n"
            "Format your answer as: \n"
            "Category: [Category Name]\n"
            "Description: [Detailed Description]"
            "For example, if the object is a closed wooden door with a glass panel, the category should be 'door' and the description should be 'a closed wooden door with a glass panel'.\n"
            "For example, if the object is a pillow with a floral pattern, the category should be 'pillow' and the description should be 'a pillow with a floral pattern'.\n"
            "For example, if the object is a wooden table, the category should be 'table' and the description should be 'a wooden table'.\n"
            "For example, if the object is a gray wall, the category should be 'wall' and the description should be 'a gray wall'.\n"
        )
        
        # response = vlm.generate_content(image_pil, prompt)
        response = "Category: nothing, Description: nothing"
        
        # Parse
        category = "Unknown"
        description = response
        if "Category:" in response and "Description:" in response:
            try:
                parts = response.split("Description:")
                cat_part = parts[0].split("Category:")[1].strip()
                desc_part = parts[1].strip()
                category = cat_part
                description = desc_part
            except:
                pass
        # Clean up category
        category = category.rstrip('.')
        obj['category'] = category
        obj['description'] = description
        
        # Set Camera Info (copied from original save_objects logic)
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., best_view_idx].detach())
        curr_cam_tran = params['cam_trans'][..., best_view_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        obj['best_view_w2c'] = curr_w2c

        llava_time_end = time.time()
        print(f"Object {i} - Category: {obj['category']}, Description: {description}")
        print(f"LLaVA Time: {llava_time_end - llava_time_start:.4f}s")

    objects = objects.to_serializable()
    objects_save_path = Path(output_dir) / "objects.pkl.gz"
    objects_save_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(objects_save_path, "wb") as f:
        pickle.dump(objects, f)
    object_time_end = time.time()

    # --- Phase 2: Relations ---
    # Use a blank image for text-only relation reasoning (or maybe obj1's crop?)
    # Since LLaVA expects image, we can use a blank one or reuse one. 
    # To be safe and fast, use a small blank image.
    relation_time_start = time.time()
    relations = []
    blank_image = Image.new('RGB', (224, 224), color='black')
    for i, obj1 in enumerate(tqdm(objects, desc="Analyzing Relations")):
        for j, obj2 in enumerate(objects[i + 1:], start=i + 1):
            llava_time_start = time.time()
            rel = {}
            rel['obj1_id'] = obj1['idx']
            rel['obj2_id'] = obj2['idx']
            
            # Construct Prompt
            prompt = (
                f"Analyze the relationship between two objects.\n"
                f"Object 1 is: {obj1['category']} ({obj1['description']}).\n"
                f"Object 2 is: {obj2['category']} ({obj2['description']}).\n"
                "Only describe the potential functional or usage relationship "
                "between object 1 and object 2.\n"
                "Focus on how object 1 can be used with or acts on object 2.\n"
                "Do NOT describe their appearance, color, material, or shape.\n"
                "For example, if object 1 is a knife and object 2 is an apple, say that the knife can be used to cut the apple. Return the verb 'cut'.\n "
                "For example, if object 1 is a chair and object 2 is a table, say that the chair can be used to sit at the table. Return the verb 'sit at'.\n" 
                "For example, if object 1 is a chair and object 2 is a window, say that the chair can be placed for looking out the window. Return the verb 'placed for looking out'.\n"
                "If there is no clear , answer exactly: "
                "'no clear'.\n"
                "Answer in ONE short English Phrase within 10 words."
            )
            
            # Generate
            # response = vlm.generate_content(blank_image, prompt)

            # test
            response = ""


            rel['pretential_relation'] = response
            relations.append(rel)
            llava_time_end = time.time()
            print(f"Relation {obj1['category']}: ({obj1['idx']}) - {obj2['category']}:({obj2['idx']}): {response}")
            print(f"LLaVA Time: {llava_time_end - llava_time_start:.4f}s")
            
    # Save Relations
    relations_save_path = Path(output_dir) / "relations.pkl.gz"
    relations_save_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(relations_save_path, "wb") as f:
        pickle.dump(relations, f)
    total_time_end = time.time()
    relation_time_end = time.time()

    print(f"Saved relations to {relations_save_path}")
    print(f"Saving map objects to: {objects_save_path}")

    print(f"Total Time - Save Objects: {object_time_end - object_time_start:.4f}s")
    print(f"Total Time - Save Relations: {relation_time_end - relation_time_start:.4f}s")
    print(f"Total Time - Save Objects & Relations: {total_time_end - total_time_start:.4f}s")


def load_data(file_path: Path) -> list:
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded from {file_path}")
    return data


def denoise_and_filter_objects(params, objects, curr_objects_idx, cfg):
    to_remove = []
    for obj in objects:
        indices = np.where(params['object_idx'] == obj['idx'])[0]

        if indices.shape[0] < 100:
            to_remove.append(obj['idx'])
            continue
    
    objects = [obj for obj in objects if obj['idx'] not in to_remove]
    objects = MapObjectList(objects)


    for curr_object_idx in curr_objects_idx:
        indices = torch.from_numpy(np.where(params['object_idx'] == curr_object_idx)[0]).to('cuda')
        object_pcd_np = params['means3D'][indices].detach().cpu().numpy()
        object_pcd_col_np = params['rgb_colors'][indices].detach().cpu().numpy()
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_pcd_np)
        object_pcd.colors = o3d.utility.Vector3dVector(object_pcd_col_np)
        cl, ind = object_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
        
        # 获取离群点的索引
        outlier_indices = np.setdiff1d(np.arange(len(object_pcd.points)), np.array(ind))
        
        # 转换离群点索引为原始索引
        outlier_indices_original = indices[outlier_indices].detach().cpu()
        
        # 将离群点的 object_idx 设置为 0
        params['object_idx'][outlier_indices_original] = 0


    # # denoise
    # for curr_object_idx in curr_objects_idx:
    #     indices = torch.from_numpy(np.where(params['object_idx'] == curr_object_idx)[0]).to('cuda')
    #     object_pcd_np = params['means3D'][indices].detach().cpu().numpy()
    #     object_pcd_col_np = params['rgb_colors'][indices].detach().cpu().numpy()
    #     object_pcd = o3d.geometry.PointCloud()
    #     object_pcd.points = o3d.utility.Vector3dVector(object_pcd_np)
    #     object_pcd.colors = o3d.utility.Vector3dVector(object_pcd_col_np)

    #     object_pcd_clusters = object_pcd.cluster_dbscan(eps=1.0, min_points=10,)
    #     object_pcd_clusters = np.array(object_pcd_clusters)
    #     counter = Counter(object_pcd_clusters)

    #     # Remove the noise label
    #     if counter and (-1 in counter):
    #         del counter[-1]

    #     if counter:
    #         # Find the label of the largest cluster
    #         most_common_label, _ = counter.most_common(1)[0]
            
    #         # Create mask for points in the largest cluster
    #         invaild_mask = object_pcd_clusters != most_common_label

    #         # Apply mask
    #         outlier_indices_original = indices[invaild_mask].detach().cpu().numpy()
    #         params['object_idx'][outlier_indices_original] = 0         



    
    return params


def regular_objects(objects: MapObjectList, cfg):
    pass

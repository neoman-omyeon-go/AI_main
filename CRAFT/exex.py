import torch
import cv2
import numpy as np
from pathlib import Path
from glob import glob
import json
import os
from collections import OrderedDict
from craft import CRAFT
from refinenet import RefineNet
import craft_utils
import imgproc
import file_utils

def load_model(yolo_weights, craft_weights, use_cuda):
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights)

    craft_model = CRAFT()
    
    if use_cuda:
        craft_model = craft_model.cuda()
        yolo_model = yolo_model.cuda()
        craft_state_dict = torch.load(craft_weights)
    else:
        craft_state_dict = torch.load(craft_weights, map_location=torch.device('cpu'))
    
    new_state_dict = OrderedDict()
    for k, v in craft_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    craft_model.load_state_dict(new_state_dict)
    craft_model.eval()

    return yolo_model, craft_model

def detect_objects(model, image_path):
    results = model(image_path)
    results.save()
    return results.xyxy[0]

def detect_text(craft_model, image, use_cuda):
    image = imgproc.loadImage(image)
    bboxes, polys, _ = test_net(craft_model, image, 0.7, 0.4, 0.4, use_cuda, False)
    return polys

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, _ = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    boxes, polys = craft_utils.getDetBoxes(score_text, y[0, :, :, 1].cpu().data.numpy(), text_threshold, link_threshold, low_text, poly)
    boxes = craft_utils.adjustResultCoordinates(boxes, 1 / target_ratio, 1 / target_ratio)
    polys = craft_utils.adjustResultCoordinates(polys, 1 / target_ratio, 1 / target_ratio)
    return boxes, polys, None

def main(image_path):
    yolo_weights = '/home/eslab/osh/CapStone/yolo/yolov5/runs/train/exp/weights/best.pt'
    craft_weights = '/home/eslab/osh/CapStone/OCR/CRAFT-pytorch/craft_mlt_25k.pth'
    use_cuda = torch.cuda.is_available()

    yolo_model, craft_model = load_model(yolo_weights, craft_weights, use_cuda)

    bboxes = detect_objects(yolo_model, image_path)
    print(f"Detected {len(bboxes)} objects in {image_path}")

    for bbox in bboxes:
        x1, y1, x2, y2, conf, cls = bbox
        crop_img = cv2.imread(image_path)[int(y1):int(y2), int(x1):int(x2)]
        polys = detect_text(craft_model, crop_img, use_cuda)
        print(f"Detected {len(polys)} text regions")

if __name__ == '__main__':
    image_path = input("input_path_image : ")
    main(image_path)

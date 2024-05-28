import os
import sys
from glob import glob as gb
from pathlib import Path
import torch

# 이전에 정의한 모듈들을 임포트
from .detect3 import run
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (
    LOGGER,
    check_requirements,
    cv2,
    non_max_suppression,
    print_args,
    increment_path
)
from utils.torch_utils import select_device

# 경로와 루트 설정
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# default_source = '/home/eslab/osh/CapStone/'
# parser.add_argument("--source", type=str, help="file/dir/URL/glob/screen/0(webcam)", default=default_source) # default_source

def prepare_options(image_path,output_path):
    """ 설정값을 직접 생성하는 함수 """
    image_files = '/home/eslab/osh/CapStone/'

    opt = {
        "weights": '/home/eslab/osh/CapStone_last/detection/yolo/yolov5/runs/train/exp/weights/best.pt',
        "source": image_path,
        "data": ROOT / "data/coco128.yaml",
        # "data": "data/coco128.yaml",
        "imgsz": [416, 416],
        "conf_thres": 0.5,
        "iou_thres": 0.45,
        "max_det": 1000,
        "device": "",
        "view_img": False,
        "save_txt": False,
        "save_csv": False,
        "save_conf": True,
        "save_crop": True,
        "nosave": False,
        "classes": None,
        "agnostic_nms": True,
        "augment": False,
        "visualize": False,
        "update": False,
        "project": output_path,
        # "project": ROOT / output_path,
        "name": output_path,
        "exist_ok": True,
        "line_thickness": 3,
        "hide_labels": False,
        "hide_conf": False,
        "half": False,
        "dnn": False,
        "vid_stride": 1
    }
    return opt

def object_detection(image_path,output_path):
    """ 주요 실행 함수 """
    # image_path = input("input_path_image : ")
    opt = prepare_options(image_path, output_path)
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    results = run(**opt)





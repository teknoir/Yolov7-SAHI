#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:14:42 2024

@author: felix
"""
import os
import sys
import json
import time
import math
import base64
import logging
from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.utils.cv import read_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tracker.byte_tracker import BYTETracker

from datetime import timezone, datetime


def error_str(rc):
    return '{}: {}'.format(rc, mqtt.error_string(rc))


def on_connect_v3(client, _userdata, _flags, rc):
    logger.info('Connected to MQTT broker {}'.format(error_str(rc)))
    if rc == 0:
        client.subscribe(args['MQTT_IN_0'], qos=0)


def on_connect_v5(client, _userdata, _flags, rc, _props):
    logger.info('Connected to MQTT broker {}'.format(error_str(rc)))
    if rc == 0:
        client.subscribe(args['MQTT_IN_0'], qos=0)


# def base64_encode(ndarray_image):
#     buff = BytesIO()
#     Image.fromarray(ndarray_image).save(buff, format='JPEG')
#     string_encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
#     return f"data:image/jpeg;base64,{string_encoded}"


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj,):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.array):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
def detection_result2ByteTrack(result):
    out = []
    names = []
    for r in result.object_prediction_list:
        b_result = r.bbox.to_xyxy()
        b_result.append(r.score.value)
        b_result.append(r.category.id)
        out.append(b_result)
        names.append(r.category.name)
    return np.array(out),names
        

APP_NAME = os.getenv('APP_NAME', 'yolov7-SAHI-bytetrack')
APP_VERSION = os.getenv('APP_VERSION', '0.1.0')

args = {
    "APP_NAME": APP_NAME,
    "APP_VERSION": APP_VERSION,
    'MQTT_IN_0': os.getenv("MQTT_IN_0",   f"{APP_NAME}/images"),
    'MQTT_OUT_0': os.getenv("MQTT_OUT_0", f"{APP_NAME}/events"),
    'MQTT_VERSION': os.getenv("MQTT_VERSION", '3'),
    'MQTT_TRANSPORT': os.getenv("MQTT_TRANSPORT", 'tcp'),
    'MQTT_SERVICE_HOST': os.getenv('MQTT_SERVICE_HOST', '127.0.0.1'),
    'MQTT_SERVICE_PORT': int(os.getenv('MQTT_SERVICE_PORT', '1883')),
    'DEVICE': os.getenv("DEVICE", '0'),
    'WEIGHTS': str(os.getenv("WEIGHTS", "model.pt")),
    'AGNOSTIC_NMS': os.getenv("AGNOSTIC_NMS", ""),
    'IMG_SIZE': int(os.getenv("IMG_SIZE", "640")),
    'CLASS_NAMES': os.getenv("CLASS_NAMES", "/home/felix/obj.names"),
    'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", "0.45")),
    'CONF_THRESHOLD': float(os.getenv("CONF_THRESHOLD", "0.25")),
    'AUGMENTED_INFERENCE': os.getenv("AUGMENTED_INFERENCE", ""),
    'CLASSES_TO_DETECT': str(os.getenv("CLASSES_TO_DETECT", "person,bicycle,car,motorbike,truck")),
    "TRACKER_THRESHOLD": float(os.getenv("TRACKER_THRESHOLD", "0.5")),
    "TRACKER_MATCH_THRESHOLD": float(os.getenv("TRACKER_MATCH_THRESHOLD", "0.8")),
    "TRACKER_BUFFER": int(os.getenv("TRACKER_BUFFER", "30")),
    "TRACKER_FRAME_RATE": int(os.getenv("TRACKER_FRAME_RATE", "10")),
    "ENABLE_SAHI" : os.getenv("ENABLE_SAHI","")
}

logger = logging.getLogger(args['APP_NAME'])
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

logger.info("TΞꓘN01R")
logger.info("TΞꓘN01R")
logger.info("TΞꓘN01R")

logger.info(json.dumps(args))





if args["AUGMENTED_INFERENCE"] == "" or args["AUGMENTED_INFERENCE"].lower() == "false":
    args["AUGMENTED_INFERENCE"] = False
else:
    args["AUGMENTED_INFERENCE"] = True

if args["ENABLE_SAHI"] == "" or args["ENABLE_SAHI"].lower() == "false":
    args["ENABLE_SAHI"] = False
else:
    args["ENABLE_SAHI"] = True

if args["AGNOSTIC_NMS"] == "" or args["AGNOSTIC_NMS"].lower() == "false":
    args["AGNOSTIC_NMS"] = False
else:
    args["AGNOSTIC_NMS"] = True

if args["CLASS_NAMES"] != "":
    class_names = []
    with open(args["CLASS_NAMES"], "r", encoding='utf-8') as names_file:
        for line in names_file:
            if line != "" and line != "\n":
                class_names.append(line.strip())
    args["CLASS_NAMES"] = class_names
else:
    logger.error("You must specify 'CLASS_NAMES'")
    sys.exit(1)

if args["CLASSES_TO_DETECT"] == "":
    args["CLASSES_TO_DETECT"] = None
else:
    cls_to_detect = args["CLASSES_TO_DETECT"]
    if len(cls_to_detect) == 1:
        cls_to_detect = args["CLASS_NAMES"].index(cls_to_detect)
    else:
        cls_to_detect = cls_to_detect.split(",")
        cls_ids = []
        for index, cls_name in enumerate(cls_to_detect):
            cls_id = args["CLASS_NAMES"].index(cls_name)
            cls_ids.append(cls_id)
        args["CLASSES_TO_DETECT"] = cls_ids
        del cls_ids, cls_to_detect


logger.info("Loading YOLOv7 Model")

yolov7_model_path = '/home/felix/model.pt'

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov7pip', # or 'yolov7hub'
    model_path=yolov7_model_path,
    confidence_threshold=0.3,
    device="cuda:0", # or 'cuda:0'
)


tracker = BYTETracker(track_thresh=args["TRACKER_THRESHOLD"],
                      match_thresh=args["TRACKER_MATCH_THRESHOLD"],
                      track_buffer=args["TRACKER_BUFFER"],
                      frame_rate=args["TRACKER_FRAME_RATE"])

myImage = plt.imread('/mnt/e/data/teknoir/test.png')[:,:,0:3]
myImagePIL = Image.fromarray((myImage[:,:,::-1]*255).astype(np.uint8))

result = get_prediction(myImagePIL, detection_model)
img_height, img_width = result.image_height,result.image_width

out,names =detection_result2ByteTrack(result)

tracked_objects = tracker.update(raw_detection)
result_sliced = get_sliced_prediction(
    myImagePIL,
    detection_model,
    slice_height = 640,
    slice_width = 384,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2,
)
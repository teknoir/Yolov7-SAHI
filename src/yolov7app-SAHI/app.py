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
from io import BytesIO

from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.utils.cv import read_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tracker.byte_tracker import BYTETracker
import paho.mqtt.client as mqtt
from torchvision.ops import nms
from torch import tensor

from datetime import timezone, datetime

def load_image(base64_image):
    image_base64 = base64_image.split(',', 1)[-1]
    image = Image.open(BytesIO(base64.b64decode(image_base64)))
    im0 = np.array(image)
    height = im0.shape[0]
    width = im0.shape[1]
    return im0, height, width

def error_str(rc):
    return '{}: {}'.format(rc, mqtt.error_string(rc))

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
    
# select only the classes we specified and process raw model outputs for ByteTrack
def detection_result2ByteTrack(result,class_selection=None):
    out = []
    names = []
    out = np.empty((0,6), float)

    for r in result.object_prediction_list:
        b_result = r.bbox.to_xyxy()
        b_result.append(r.score.value)
        b_result.append(r.category.id)
        if class_selection is not None:
            #logger.info("{:} {:}".format(class_selection,r.category.name))
            if r.category.id in class_selection:
                out = np.concatenate((out, [b_result]))
                names.append(r.category.name)
        else:
            out = np.concatenate((out, [b_result]))
            names.append(r.category.name)
    return out,names
        
    
class yolov7app_SAHI():
    def on_connect_v3(self,client, _userdata, _flags, rc):
        self.logger.info('Connected to MQTT broker {}'.format(error_str(rc)))
        if rc == 0:
            self.client.subscribe(self.args['MQTT_IN_0'], qos=0)


    def on_connect_v5(self, client,_userdata, _flags, rc, _props):
        self.logger.info('Connected to MQTT broker {}'.format(error_str(rc)))
        if rc == 0:
            self.client.subscribe(self.args['MQTT_IN_0'], qos=0)
       
    def detect_and_track(self,im0):
        t0 = time.perf_counter()
        # if SAHI is enabled, use the sliding window approach
        if self.args["ENABLE_SAHI"]:
            result = get_sliced_prediction(
                im0,
                self.model,
                slice_height = self.args['SLICE_WIDTH'],
                slice_width = self.args['SLICE_HEIGHT'],
                overlap_height_ratio = self.args['OVERLAP_HEIGHT'],
                overlap_width_ratio = self.args['OVERLAP_WIDTH'],
                verbose = 0
            )
        else:
            result = get_prediction(im0, self.model)
        raw_detection,names = detection_result2ByteTrack(result,class_selection=self.args["CLASSES_TO_DETECT"])
        
        # do non-max suppression filtering
        keep = nms(tensor(raw_detection[:,0:4]),tensor(raw_detection[:,4]),self.args['IOU_THRESHOLD'])
        
        # keep only the filtered entries
        raw_detection = raw_detection[np.array(keep),:]
        
        #update the tracker
        tracked_objects = self.tracker.update(raw_detection)
        inference_time = time.perf_counter()-t0
        t = result.durations_in_seconds
        if not 'slice' in t.keys():
            t['slice'] = 0.0
        self.logger.info("{} Objects - Time: {:3.1f} ms slice: {:3.1f} ms inference: {:3.1f} ms".format(
            len(tracked_objects), inference_time*1e3,t['slice']*1e3,t['prediction']*1e3))
        return tracked_objects,raw_detection


    def on_message(self,c, userdata, msg):
        message = str(msg.payload.decode("utf-8", "ignore"))
        # payload: {“timestamp”: “…”, “image”: <base64_mime>, “camera_id”: “A”, “camera_name”: “…”}
        
        try: 
            data_received = json.loads(message)
        except json.JSONDecodeError as e:
            self.logger.error("Error decoding JSON:", e)
            return
        
        if "image" not in data_received:
            self.logger.error("No Image. Exiting.")
            return
        
        if "location" not in data_received:
            self.logger.warning("No Location. Proceeding.")
            data_received["location"] = {"country": "",
                                         "region": "",
                                         "site": "",
                                         "zone": "",
                                         "group": ""}
        
        if "timestamp" not in data_received:
            self.logger.warning("No timestamp. Using current time.")
            data_received["timestamp"] = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

        msg_time_0 = time.perf_counter()

        try:
            img, orig_height, orig_width = load_image(data_received["image"])
        except Exception as e:
            self.logger.error(f"Could not load image. Error: {e}")
            return
        
        tracked_objects,raw_detection = self.detect_and_track(img)

        runtime = time.perf_counter() - msg_time_0

        base_payload = {
                "timestamp": data_received["timestamp"],
                "location": data_received["location"],
                "type": "object"
            }

        if "peripheral" in data_received:
            base_payload["peripheral"] = data_received["peripheral"]

        if "lineage" in data_received:
            base_payload["lineage"] = data_received["lineage"]
        else:
            base_payload["lineage"] = []

        base_payload["lineage"].append({"name": self.args['APP_NAME'],
                                        "version": self.args['APP_VERSION'], 
                                        "runtime": runtime})

        detections = []
        for trk in tracked_objects:
            detection_event = base_payload.copy()

            obj = {}
            obj["id"] = str(trk[4])
            obj["x1"] = float(int(trk[0]) / orig_width)
            obj["y1"] = float(int(trk[1]) / orig_height)
            obj["x2"] = float(int(trk[2]) / orig_width)
            obj["y2"] = float(int(trk[3]) / orig_height)
            obj["width"] = float(obj["x2"] - obj["x1"])
            obj["height"] = float(obj["y2"] - obj["y1"])
            obj["area"] = float(obj["height"] * obj["width"])
            obj["ratio"] = float(obj["height"] / obj["width"])
            obj["x_center"] = float((obj["x1"] + obj["x2"])/2.)
            obj["y_center"] = float((obj["y1"] + obj["y2"])/2.)
            obj["score"] = float(trk[6])
            obj["class_id"] = int(trk[5])
            obj["label"] = self.args["CLASS_NAMES"][obj["class_id"]]

            detection_event["detection"] = obj

            detections.append(detection_event)


        output = base_payload.copy() # copy everything for frontend, even if no detections
        output["detections"] = detections
        output["image"] = data_received["image"]

        msg = json.dumps(output, cls=NumpyEncoder)
        self.client.publish(userdata['MQTT_OUT_0'], msg)
        
    def __init__(self,args,logger):
        self.logger = logger
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

        if args['DEVICE']=="0":
            device = "cuda:0"
        else:
            device = "cpu"
        logger.info("Loading YOLOv7 Model")
        
        # init arguments
        self.args = args

        # setup a model
        self.model = AutoDetectionModel.from_pretrained(
            model_type='yolov7pip', # or 'yolov7hub'
            model_path=args['WEIGHTS'],
            confidence_threshold=args['CONF_THRESHOLD'],
            device=device, # or 'cuda:0'
            image_size=args['IMG_SIZE']
        )
        self.tracker = BYTETracker(track_thresh=args["TRACKER_THRESHOLD"],
                              match_thresh=args["TRACKER_MATCH_THRESHOLD"],
                              track_buffer=args["TRACKER_BUFFER"],
                              frame_rate=args["TRACKER_FRAME_RATE"])
        if args['MQTT_VERSION'] == '5':
            self.client = mqtt.Client(client_id=args['APP_NAME'],
                                 transport=args['MQTT_TRANSPORT'],
                                 protocol=mqtt.MQTTv5,
                                 userdata=self.args)
            self.client.reconnect_delay_set(min_delay=1, max_delay=120)
            self.client.on_connect = self.on_connect_v5
            self.client.on_message = self.on_message
            self.client.connect(args['MQTT_SERVICE_HOST'], port=args['MQTT_SERVICE_PORT'],
                           clean_start=mqtt.MQTT_CLEAN_START_FIRST_ONLY, keepalive=60)

        if args['MQTT_VERSION'] == '3':
            self.client = mqtt.Client(client_id=args['APP_NAME'], transport=args['MQTT_TRANSPORT'],
                                 protocol=mqtt.MQTTv311, userdata=self.args, clean_session=True)
            self.client.reconnect_delay_set(min_delay=1, max_delay=120)
            self.client.on_connect = self.on_connect_v3
            self.client.on_message = self.on_message
            self.client.connect(args['MQTT_SERVICE_HOST'],
                           port=args['MQTT_SERVICE_PORT'], keepalive=60)

        self.client.enable_logger(logger=self.logger)
    
    def run(self):
        # This runs the network code in a background thread and also handles reconnecting for you.

        self.client.loop_forever()
        
        







    



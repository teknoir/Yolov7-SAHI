#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:14:42 2024

@author: felix
"""

from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.utils.cv import read_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tracker.byte_tracker import BYTETracker

from collections import namedtuple
yolov7_model_path = '/home/felix/model.pt'

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
        
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov7pip', # or 'yolov7hub'
    model_path=yolov7_model_path,
    confidence_threshold=0.3,
    device="cuda:0", # or 'cuda:0'
)


TrackerArg = namedtuple('TrackerArg', 'track_thresh track_buffer')
arg = TrackerArg(track_thresh=0.1,track_buffer=600)

tracker = BYTETracker(arg)

myImage = plt.imread('/mnt/e/data/teknoir/test.png')[:,:,0:3]
myImagePIL = Image.fromarray((myImage[:,:,::-1]*255).astype(np.uint8))

result = get_prediction(myImagePIL, detection_model)
img_height, img_width = result.image_height,result.image_width

out,names =detection_result2ByteTrack(result)

tracker.update(out[:,0:5], [img_height, img_width], [img_height, img_width])
result_sliced = get_sliced_prediction(
    myImagePIL,
    detection_model,
    slice_height = 640,
    slice_width = 384,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2,
)
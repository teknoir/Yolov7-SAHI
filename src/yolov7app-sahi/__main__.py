"""Command-line interface."""
import click
import os
import sys
import logging
import json
from app import yolov7app_SAHI
@click.command()
@click.version_option()
def main() -> None:
    """YOLOv7 with SAHI."""
    APP_NAME = os.getenv('APP_NAME', 'yolov7-SAHI-bytetrack')
    APP_VERSION = os.getenv('APP_VERSION', '0.1.0')

    args = {
        "APP_NAME": APP_NAME,
        "APP_VERSION": APP_VERSION,
        'MQTT_IN_0': os.getenv("MQTT_IN_0",   f"{APP_NAME}/images"),
        'MQTT_OUT_0': os.getenv("MQTT_OUT_0", f"{APP_NAME}/detections"),
        'MQTT_VERSION': os.getenv("MQTT_VERSION", '3'),
        'MQTT_TRANSPORT': os.getenv("MQTT_TRANSPORT", 'tcp'),
        'MQTT_SERVICE_HOST': os.getenv('MQTT_SERVICE_HOST', '127.0.0.1'),
        'MQTT_SERVICE_PORT': int(os.getenv('MQTT_SERVICE_PORT', '1883')),
        'DEVICE': os.getenv("DEVICE", '0'),
        'WEIGHTS': str(os.getenv("WEIGHTS", "model.pt")),
        'AGNOSTIC_NMS': os.getenv("AGNOSTIC_NMS", ""),
        'IMG_SIZE': int(os.getenv("IMG_SIZE", "640")),
        'CLASS_NAMES': os.getenv("CLASS_NAMES", "obj.names"),
        'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", "0.45")),
        'CONF_THRESHOLD': float(os.getenv("CONF_THRESHOLD", "0.25")),
        'AUGMENTED_INFERENCE': os.getenv("AUGMENTED_INFERENCE", ""),
        'CLASSES_TO_DETECT': str(os.getenv("CLASSES_TO_DETECT", "person,bicycle,car,motorbike,truck")),
        "TRACKER_THRESHOLD": float(os.getenv("TRACKER_THRESHOLD", "0.5")),
        "TRACKER_MATCH_THRESHOLD": float(os.getenv("TRACKER_MATCH_THRESHOLD", "0.8")),
        "TRACKER_BUFFER": int(os.getenv("TRACKER_BUFFER", "100")),
        "TRACKER_FRAME_RATE": int(os.getenv("TRACKER_FRAME_RATE", "10")),
        "ENABLE_SAHI" : os.getenv("ENABLE_SAHI",""),
        "SLICE_WIDTH" :  int(os.getenv("SLICE_WIDTH","640")),
        "SLICE_HEIGHT" :  int(os.getenv("SLICE_HEIGHT","384")),
        "OVERLAP_WIDTH" :  float(os.getenv("OVERLAP_WIDTH","0.2")),
        "OVERLAP_HEIGHT" :  float(os.getenv("OVERLAP_HEIGHT","0.2")),
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
    myDetector = yolov7app_SAHI(args,logger)
    myDetector.run()


if __name__ == "__main__":
    main(prog_name="yolov7app-SAHI")  # pragma: no cover

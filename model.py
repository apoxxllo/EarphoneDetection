# Import necessary libraries
from roboflow import Roboflow
import torch
import cv2
import numpy as np

# Initialize Roboflow and download dataset
rf = Roboflow(api_key="Irku5t4OOrput5J405G6")
project = rf.workspace("pcv-fp").project("fp-pcv-qwpxl")
version = project.version(4)
dataset = version.download("yolov5")

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt')

# Function to run object detection
def detect_objects(image):
    results = model(image)
    return results

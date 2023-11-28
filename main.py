import cv2
import requests
import numpy as np

from ultralytics import YOLO
# model = YOLO('yolov5nu.pt')
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
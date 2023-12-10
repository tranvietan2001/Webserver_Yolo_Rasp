import requests
import numpy as np
import cv2
from ultralytics import YOLO

ip = "http://192.168.1.7:5000"
url_status = ip+"/status"
url_img = ip+"/data_img"

model = YOLO('yolov5nu.pt')

while True:

    resp_img = requests.get(url_img)
    # resp_status = requests.get(url_status)
       
    if resp_img.status_code == 200 :
        image_array = np.asarray(bytearray(resp_img.content), dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        cv2.imshow("W", frame)    

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        print("Error IMG:", resp_img.status_code)
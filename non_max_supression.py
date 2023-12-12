import requests
import numpy as np
import cv2
import torch

from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression


ip = "http://192.168.1.7:5000"
url = ip+"/status"
url_img = ip+"/data_img"

model = YOLO('yolov5nu.pt')
cap = cv2.VideoCapture(0)


prediction = []

while True:

    # response2 = requests.get(url_img)
    # image_array = np.asarray(bytearray(response2.content), dtype=np.uint8)
    # frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    r,frame = cap.read()
    
    center_def_x = int(640/2)
    center_def_y = int(480/2)
    point_center = (center_def_x, center_def_y)
    box_x_tl = 220
    box_y_tl = 140
    box_x_br = 420
    box_y_br = 340
    point_topleft = (box_x_tl, box_y_tl)
    point_bottomright = (box_x_br, box_y_br)
    frame = cv2.circle(frame, point_center, 2, (0,255,255), 5)
    frame = cv2.rectangle(frame, point_topleft, point_bottomright, (0,255,255), 2)


    results = model.track(frame, persist=True)
    frame_ = results[0].plot()
    result = results[0]

    for i in range(len(result.boxes)):
        box = result.boxes[i]
        cords = box.xyxy[0].tolist()

        # cords = box.xywh[0].tolist()
        cords = [round(x) for x in cords]
        w = cords[2] - cords[0]
        h = cords[3] - cords[1]



        conf = round(box.conf[0].item(), 2)
        name_class = result.names[box.cls[0].item()]   

        prediction.append([cords[0], cords[1], cords[2], cords[3], conf, name_class])    

        x_center = int(cords[0]+w/2)
        y_center = int(cords[1]+h/2)
        point_center_dt = (x_center, y_center)
        # print(point_center)
        frame_ = cv2.circle(frame_, point_center_dt, 2, (0,0,255), 5)

        eccentricity_left = int(point_center[0] - point_center_dt[0])
        # eccentricity_right = int(point_center_dt[0] - point_center[0]) 
        # print(eccentricity_left, eccentricity_right)
        print(eccentricity_left)

    valid_predictions = []
    for item in prediction:
        try:
            valid_item = list(map(int, item))  # Chuyển đổi chuỗi thành số nguyên
            valid_predictions.append(valid_item)
        except ValueError:
            pass  # Bỏ qua các giá trị không hợp lệ
    prediction = torch.tensor(prediction)
    conf_thres = 0.25
    iou_thres = 0.45
    selected_indices = non_max_suppression(prediction, conf_thres=conf_thres, iou_thres=iou_thres)

    # Lấy bounding boxes và xác suất phân loại đã được lọc
    selected_bounding_boxes = [prediction[i][:4] for i in selected_indices]
    selected_confidences = [prediction[i][4] for i in selected_indices]
    selected_classes = [prediction[i][5] for i in selected_indices]     
    for bbox, confidence, cls in zip(selected_bounding_boxes, selected_confidences, selected_classes):
        print("Bounding box:", bbox)
        print("Confidence:", confidence)
        print("Class:", cls)   


    cv2.imshow('Image', frame_)

    
    # if data3 == 'E':
    #     break
    
    if cv2.waitKey(1)  & 0xFF == ord("q"):
        break
    
# cap.release()
cv2.destroyAllWindows()

           

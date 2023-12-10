import cv2
import requests
import numpy as np
import time

from ultralytics import YOLO
model = YOLO('yolov5nu.pt')

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 255, 0)  # Màu văn bản (xanh lá cây trong ví dụ này)
thickness = 2  # Độ dày của viền văn bản

x = 50
y = 50
i = 0
start_time = time.time()
while True:
    ret, frame = cap.read()


    current_time = time.time()
    elapsed_time = current_time - start_time

    print(int(elapsed_time))
    if elapsed_time >= 5:  # Dừng vòng lặp sau 5 giây
        start_time = time.time()
        print("ok")

    # if cv2.waitKey(1)  & 0xFF == ord("s"):    
    #     cv2.putText(frame, str(int(elapsed_time+1)), (x, y), font, font_scale, color, thickness)
    

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

    print(len(result.boxes)) #so luong bbx
    for i in range(len(result.boxes)):
    # #     print(i)

        box = result.boxes[i]

        #toa do box
        cords = box.xyxy[0].tolist()
        # cords = box.xywh[0].tolist()
        
        cords = [round(x) for x in cords]
        w = cords[2] - cords[0]
        h = cords[3] - cords[1]

        print(w,h)

        x_center = int(cords[0]+w/2)
        y_center = int(cords[1]+h/2)
        point_center_dt = (x_center, y_center)
        # print(point_center)
        frame_ = cv2.circle(frame_, point_center_dt, 2, (0,0,255), 5)

        eccentricity_left = int(point_center[0] - point_center_dt[0])
        # eccentricity_right = int(point_center_dt[0] - point_center[0]) 
        # print(eccentricity_left, eccentricity_right)
        print(eccentricity_left)

        if (eccentricity_left <= 10) and ( eccentricity_left >= -10):
            print("okkkkkkkkkkkk")
        



    #     class_id = result.names[box.cls[0].item()]
    #     conf = round(box.conf[0].item(), 2)
    #     box_id = box.id
        
    #     if box_id == 14:
    #         box_id =14
    #         print("-------------------------------")
    #         a = (cords[0], cords[1])
    #         b = (cords[2], cords[3])
    #         w = cords[2] - cords[0]
    #         h = cords[3] - cords[1]
    #         s = w*h
    #         frame = cv2.rectangle(frame, a, b, (255,0,0), 1)
    #         print("box: ", i, "toado: ",cords, "id_name: ", class_id,"dochinhxat: ", conf,"box_id: ", box_id,"S: ", s)
    #     # else:
    #     #     print("WARNING......")
    






    cv2.imshow("windown", frame)
    cv2.imshow("windown", frame_)

    if cv2.waitKey(1)  & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
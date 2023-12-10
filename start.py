import requests
import numpy as np
import cv2

from ultralytics import YOLO


ip = "http://192.168.1.7:5000"
url = ip+"/status"
url_img = ip+"/data_img"

model = YOLO('yolov5nu.pt')

while True:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.tex
            print("Data received:", data)

            if data == 'S':
                print("start program")


                while True:

                    response2 = requests.get(url_img)
                    image_array = np.asarray(bytearray(response2.content), dtype=np.uint8)
                    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    
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
    # #     print(i)

                        box = result.boxes[i]
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


                    cv2.imshow('Image', frame_)

                    
                    # if data3 == 'E':
                    #     break
                    
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                
                #cv2.waitKey(0)
                cv2.destroyAllWindows()

            else:
                print("=====>ERROR r1")


        else:
            print("Error:", response.status_code)


    except requests.exceptions.RequestException as e:
        print("Error:", e)

#----OK-----------



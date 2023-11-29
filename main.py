import cv2
import requests
import numpy as np

from ultralytics import YOLO
model = YOLO('yolov5nu.pt')

url = 'http://192.168.1.5 :5000/data_img'
# print(model)
# video_path = 'video/video_cut.mp4'
# video_path = 'video/test.mp4'
# cap = cv2.VideoCapture(0)
# ret = True

while True:
    # ret, frame = cap.read()

    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    # image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
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

    for i in range(1,10):
        text = i
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)  # Màu văn bản (xanh lá cây trong ví dụ này)
        thickness = 2  # Độ dày của viền văn bản

        # Vẽ văn bản lên khung hình
        cv2.putText(frame, text, (50, 50), font, font_scale, color, thickness)
    # scale_percent = 50 # percent of original size
    # width = int(frame.shape[1] * scale_percent / 100)
    # height = int(frame.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    
    # print(frame)
    results = model.track(frame, persist=True)


    frame_ = results[0].plot()
    # frame_ = results[0].plot(conf=True, line_width=False, font_size=None, font='Arial.ttf', pil=False, img=None, im_gpu=None, kpt_radius=1, kpt_line=True, labels=True, boxes=True, masks=True, probs=True)


# =======================
    result = results[0]

    # box = result.boxes[]
    # print(box.)

    print(len(result.boxes)) #so luong bbx
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
    

    cv2.imshow("Video", frame_)
    # cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

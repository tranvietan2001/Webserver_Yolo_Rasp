import cv2
import requests
import numpy as np
import time
import math
from ultralytics import YOLO


class KalmanFilter:
    def __init__(self, dt, state_dim, measurement_dim):
        self.dt = dt
        self.A = np.eye(state_dim)  # Ma trận chuyển đổi trạng thái
        self.B = np.zeros((state_dim, 1))  # Ma trận điều khiển
        self.H = np.eye(measurement_dim)  # Ma trận đo lường
        self.Q = np.eye(state_dim)  # Ma trận nhiễu quá trình
        self.R = np.eye(measurement_dim)  # Ma trận nhiễu đo lường
        self.P = np.eye(state_dim)  # Ma trận ước lượng sai số

    def predict(self, x, u=None):
        x = np.dot(self.A, x)
        if u is not None:
            x += np.dot(self.B, u)
        P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return x, P

    def update(self, x, P, z):
        y = z - np.dot(self.H, x)
        S = np.dot(np.dot(self.H, P), self.H.T) + self.R
        K = np.dot(np.dot(P, self.H.T), np.linalg.inv(S))
        x += np.dot(K, y)
        P = np.dot(np.eye(self.H.shape[0]) - np.dot(K, self.H), P)
        return x, P



#c: center, b box
def calculate_distance(point_c, point_b):
    distance = math.sqrt((point_b[0] - point_c[0])**2 + (point_b[1] - point_c[1])**2)
    return distance


model = YOLO('yolov8n.pt')
# model = YOLO('crowdhuman_yolov5m.pt')

cap = cv2.VideoCapture(0)
ip = "http://192.168.1.10:5000"
url_img = ip+"/data_img"

url_control = ip+"/data_control"
data_post = {'speed': '50', 'control': '16546,342'}
# print(tracker)

dt = 2.0  # Khoảng thời gian giữa các lần đo
state_dim = 4  # Số chiều của trạng thái (vị trí và vận tốc)
measurement_dim = 2  # Số chiều của đo lường (vị trí)
kf = KalmanFilter(dt, state_dim, measurement_dim)
predicted_positions = []



start_time = time.time()
text_time = ""
list_box = []
list_id = []

direction = 0

while True:
    # read frame
    ret, frame = cap.read()
    # response2 = requests.get(url_img)
    # image_array = np.asarray(bytearray(response2.content), dtype=np.uint8)
    # frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    #caculation time ======================
    current_time = time.time()
    elapsed_time = current_time - start_time
    text_time = str(int(elapsed_time))

    #======================================


    # draw bb center
    center_def_x = int(640/2)
    center_def_y = int(480/2)
    point_center = (center_def_x, center_def_y)  
    box_x_tl = 220
    box_y_tl = 70
    box_x_br = 420
    box_y_br = 410
    point_topleft = (box_x_tl, box_y_tl)
    point_bottomright = (box_x_br, box_y_br)
    frame = cv2.circle(frame, point_center, 2, (0,255,255), 5)
    frame = cv2.rectangle(frame, point_topleft, point_bottomright, (0,255,255), 2)
        #draw line xy 
    point_center_xy = (320,320)    
    frame = cv2.line(frame, (320,0), (320,640), (255,255,255), 2)
    frame = cv2.line(frame, (0,320), (640,320), (255,255,255), 2)
    frame = cv2.circle(frame, point_center_xy, 2, (0,255,255), 5)
    
    # tracking
    results = model.track(frame, persist=True)
    # print("RESULTS: ===> ", results)

    # detect all obj in one frame (to vsl use plot)
    frame_ = results[0].plot()
    # frame_ = frame
    
    # result detect in one frame
    result = results[0]

    
    # print("=====> bbox: ", len(result.boxes)) #so luong bb
    
    #Co so luong box->duyet lan luot qua cac box
    #->tinh diem center box detect
    #->


    if int(elapsed_time) < 15:
        list_box.clear()
        list_id.clear()
        for i in range(len(result.boxes)):

            box = result.boxes[i]
        
            # #   toa do
            # cords = box.xyxy[0].tolist()
            name_class = result.names[box.cls[0].item()]   
            conf = round(box.conf[0].item(), 2)         
            
            cord_xywh = box.xywh[0].tolist()

            if name_class == "person" and conf >= 0.5: #xét thêm đk độ chính xác bao nhiêu 
                
                #   toa do
                cords = box.xyxy[0].tolist()
                conf = round(box.conf[0].item(), 2)
                print("====> ID: ", name_class)
                print("====> CONF: ", conf)

                # print(cord_xywh)
                # print(cords)

                box_id = box.id
                if not box_id == None:
                    print("====> ID_BOX: ", box_id.item())

                    cords = [round(x) for x in cords]
                    w = cords[2] - cords[0]
                    h = cords[3] - cords[1]

                    x_center = int(cords[0]+w/2)
                    y_center = int(cords[1]+h/2)
                    point_center_dt = (x_center, y_center)

                    cal = calculate_distance(point_center, point_center_dt)

                    list_box.append(cal)
                    list_id.append(box_id.item())

                    print("list box: ", list_box)
                    print("list id: ", list_id)

                else:
                    print("=======> Not detect")

                print("=======================================================")
        
    elif int(elapsed_time) >= 15:  
        elapsed_time = 10
        print(int(elapsed_time))
        text_time = "TRACKING"
        print(list_box)

        if len(list_box) != 0:
        
            min_value = min(list_box)
            min_index = list_box.index(min_value)
            
            # id_track = box_id.item()
            id_track = list_id[min_index]
        else:
            print("Not Tracking")
            id_track = 0
        print(id_track)

        class_ids = []
        confidences = []
        boxes_ = []

        for i in range(len(result.boxes)):

            box_ = result.boxes[i]
            box_id = box_.id
            # #   toa do
            # cords = box.xyxy[0].tolist()
            name_class = result.names[box_.cls[0].item()]            

            if not box_id == None:
                print("====> ID_BOX: ", box_id.item())
                if name_class == "person" and box_id.item() == id_track:
                # if name_class == "person":
                    print("di chuyen theo huong cua doi tuong")
                    cords = box_.xyxy[0].tolist()
                
                    cords = [round(x) for x in cords]
                    w = cords[2] - cords[0]
                    h = cords[3] - cords[1]
                    x = cords[0]
                    y = cords[1]


                    x_center = int(cords[0]+w/2)
                    y_center = int(cords[1]+h/2)
                    point_center_dt = (x_center, y_center)

                    cal = calculate_distance(point_center, point_center_dt)
                    # print(cal)

                    conf = round(box_.conf[0].item(), 2)
                    
                    class_ids.append(box_id)
                    boxes_.append([x, y, w, h])
                    confidences.append(float(conf))


                    # frame_ = cv2.circle(frame_, point_center_dt, 2, (0,0,255), 5)
                    # frame_ = cv2.line(frame_, point_center, point_center_dt, (0,0,255), 5)

                    # frame_ = cv2.line(frame_, (320,0), (320,640), (255,255,255), 2)
                    # frame_ = cv2.line(frame_, (0,320), (640,320), (255,255,255), 2)
                else:
                    print("ID bị thay doi => canh bao, tnay doi sang doi tuong id gan nhat")
            else:
                print("=======> Not detect")
        
        for box in boxes_:
            x, y, w, h = box

            # Dự đoán vị trí và cập nhật bằng bộ lọc Kalman
            measurement = np.array([x + w / 2, y + h / 2])  # Đo lường vị trí trung tâm của bounding box
            if len(predicted_positions) == 0:
                initial_state = np.array([x + w / 2, y + h / 2, 0, 0])  # Trạng thái ban đầu (vị trí và vận tốc)
                kf = KalmanFilter(dt, state_dim, measurement_dim)
                predicted_state, _ = kf.predict(initial_state)
            else:
                predicted_state, _ = kf.predict(predicted_positions[-1])

            predicted_positions.append(predicted_state)
            
            # Vẽ bounding box và vị trí dự đoán lên khung hình
            cv2.rectangle(frame_, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_, "ID: ", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            point_c = (x+w/2, y+h/2)
            point_c = [round(x) for x in point_c]
            cv2.circle(frame_, point_c, 4, (0,255,0), 2)
            
            print("====>  POINT_CENTER_KL: ", point_c)           
            print(x,y,w,h)

            s  = w * h

            if (280 <= point_c[0] <= 360) and (280 <= point_c[1] <= 360) :
                direction = 1
                print("======> STOP")
            # else:
            #     print("====> right")
                
            elif (point_c[0] < 280):
                direction = 2
                print("======> RIGHT")
            elif (point_c[0] > 360):
                direction = 3
                print("======> LEFT")
            # else:
            #     direction = 1111
    frame_ = cv2.putText(frame_, text_time, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    #visualize
    cv2.imshow("windown", frame_)

    # exit   
    if cv2.waitKey(1)  & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
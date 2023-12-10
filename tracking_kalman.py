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


model = YOLO('yolov5nu.pt')
# model = YOLO('crowdhuman_yolov5m.pt')

# cap = cv2.VideoCapture(0)
ip = "http://192.168.1.4:5000"
url_img = ip+"/data_img"




# Sử dụng bộ lọc Kalman và YOLO để tracking
dt = 1.0  # Khoảng thời gian giữa các lần đo
state_dim = 4  # Số chiều của trạng thái (vị trí và vận tốc)
measurement_dim = 2  # Số chiều của đo lường (vị trí)
kf = KalmanFilter(dt, state_dim, measurement_dim)

# # Load mô hình YOLO
# net = cv2.dnn.readNetFromDarknet('path_to_config_file.cfg', 'path_to_weights_file.weights')

# # Tên các lớp trong mô hình YOLO
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Đọc video hoặc webcam
# video = cv2.VideoCapture('path_to_video_file.mp4')
video = cv2.VideoCapture(0)

# Khởi tạo biến lưu trữ vị trí dự đoán của đối tượng
predicted_positions = []

while True:
    # Đọc từng khung hình trong video
    ret, frame = video.read()

    if not ret:
        break

    # Phát hiện đối tượng bằng YOLO
    # blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # net.setInput(blob)
    # outs = net.forward(output_layers)

    results = model.track(frame, persist=True)
    
    frame = results[0].plot()
    result = results[0]


    # Xác định bounding box và vị trí trung tâm của đối tượng
    class_ids = []
    confidences = []
    boxes_ = []
    # for out in outs:
    #     for detection in out:
    #         scores = detection[5:]
    #         class_id = np.argmax(scores)
    #         confidence = scores[class_id]
    #         if confidence > 0.5:
    #             center_x = int(detection[0] * frame.shape[1])
    #             center_y = int(detection[1] * frame.shape[0])
    #             w = int(detection[2] * frame.shape[1])
    #             h = int(detection[3] * frame.shape[0])
    #             x = int(center_x - w / 2)
    #             y = int(center_y - h / 2)
    #             boxes.append([x, y, w, h])
    #             confidences.append(float(confidence))
    #             class_ids.append(class_id)

    for i in range(len(result.boxes)):
        box_ = result.boxes[i]

        cords = box_.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        w = cords[2] - cords[0]
        h = cords[3] - cords[1]
        x = cords[0]
        y = cords[1]

        conf = round(box_.conf[0].item(), 2)
        box_id = box_.id.item()
        if not box_id == None:
            class_ids.append(box_id)
        else:
            print("Not find objects")

        boxes_.append([x, y, w, h])
        confidences.append(float(conf))
        
    # Áp dụng bộ lọc Kalman để dự đừ định vị vị trí của đối tượng
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (int(predicted_state[0]), int(predicted_state[1])), 4, (0, 255, 255), -1)

    # Hiển thị khung hình kết quả
    cv2.imshow("Tracking", frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Giải phóng bộ nhớ và đóng cửa sổ
video.release()
cv2.destroyAllWindows()



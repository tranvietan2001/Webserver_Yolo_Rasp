import cv2
import numpy as np

# Tạo bộ lọc Kalman
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

# Khởi tạo biến lưu trữ vị trí hiện tại của đối tượng
current_measurement = np.array((2, 1), np.float32)
current_prediction = np.zeros((2, 1), np.float32)

# Đọc video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Thực hiện YOLO detection để xác định vị trí của đối tượng
    # (giả sử được lưu trong biến `object_position` dạng (x, y))

    # Cập nhật bộ lọc Kalman
    kalman.correct(current_measurement)

    # Dự đoán vị trí tiếp theo của đối tượng
    current_prediction = kalman.predict()

    # Cập nhật vị trí hiện tại của đối tượng
    current_measurement[0] = object_position[0]
    current_measurement[1] = object_position[1]

    # Vẽ vị trí dự đoán của đối tượng
    predicted_x = int(current_prediction[0])
    predicted_y = int(current_prediction[1])
    cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 255, 0), -1)

    # Vẽ vị trí hiện tại của đối tượng
    cv2.circle(frame, (object_position[0], object_position[1]), 5, (0, 0, 255), -1)

    # Hiển thị khung hình kết quả
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng video
cap.release()
cv2.destroyAllWindows()
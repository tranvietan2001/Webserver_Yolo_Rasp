import cv2
import threading
import requests
import numpy as np

ip = 'http://192.168.16.117:5000'
url_came = ip+'/data_img'
url_status = ip+'/status'

# Biến cờ để kiểm soát việc chạy luồng camera
camera_running = False

# Hàm để hiển thị hình ảnh từ camera
def show_image():
    cap = cv2.VideoCapture(0)  # Mở camera
    while camera_running:
        ret, frame = cap.read()
        cv2.imshow('Camera', frame)  # Hiển thị hình ảnh từ camera
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def show_webcame():
    while camera_running:

        resp_came = requests.get(url_came)
        image_array = np.asarray(bytearray(resp_came.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        cv2.imshow('Image', image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            resp_came.close()
            break

    cv2.destroyAllWindows()

# Hàm để lắng nghe tín hiệu UART
def listen_even():
    global camera_running
    while True:

        try:
            response = requests.get(url_status)
            if response.status_code == 200:
                data = response.text
                print("Data received:", data)

                if data == "S":
                    print("===> Open camera thread")
                    if not camera_running:
                        camera_running = True
                        # camera_thread = threading.Thread(target=show_image)
                        camera_thread = threading.Thread(target=show_webcame)
                        camera_thread.start()  # Khởi động luồng camera
                elif data == "E":
                    print("===> Close camera thread.......")
                    if camera_running:
                        camera_running = False
                        camera_thread.join()  # Dừng luồng camera nếu đang chạy
                    
            else:
                print("Error:", response.status_code)
        except requests.exceptions.RequestException as e:
            print("Error:", e)



       

# Tạo luồng cho việc lắng nghe resq
active_ev = threading.Thread(target=listen_even)
active_ev.start()  # Khởi động luồng lắng nghe UART

# Chờ cho đến khi luồng lắng nghe UART kết thúc
active_ev.join()

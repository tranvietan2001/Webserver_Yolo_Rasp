import cv2

cap = cv2.VideoCapture(0)

# Thiết lập thông số cho văn bản
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 255, 0)  # Màu văn bản (xanh lá cây trong ví dụ này)
thickness = 2  # Độ dày của viền văn bản

# Vị trí ban đầu của văn bản
x = 50
y = 50
i = 0
while True:
    ret, frame = cap.read()

    a = i/10
    cv2.putText(frame, str(a), (x, y), font, font_scale, color, thickness)
    
    cv2.imshow("windown", frame)
    i = i + 1 

    if a >=10:
        a = 10

    if cv2.waitKey(1)  & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
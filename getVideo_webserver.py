import requests
import numpy as np
import cv2

url = 'http://192.168.1.3:5000/data_img'

while True:

    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    cv2.imshow('Image', image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
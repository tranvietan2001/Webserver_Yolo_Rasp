import requests
import numpy as np
import cv2

ip = 'http://192.168.16.117:5000'
url = ip+'/data_img'

while True:

    response = requests.get(url)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        cv2.imshow('Image', image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            response.close()
            break
            
    else:
        print("Error:", response.status_code)
   
cv2.destroyAllWindows()
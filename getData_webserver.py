import requests

# url = "http://192.168.1.11/data"

url = "http://192.168.16.117:5000/status"

while True:
 
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text
            print("Data received:", data)
            
        else:
            print("Error:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("Error:", e)

#----OK-----------



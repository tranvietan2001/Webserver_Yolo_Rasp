import requests

url = 'http://192.168.1.5:5000/data_control'  # Thay thế <ip_address> bằng địa chỉ IP cụ thể
data = {'speed': '50', 'control': '16546,342'}  # Dữ liệu bạn muốn gửi

while True: 
    response = requests.post(url, data=data)

    if response.status_code == 200:
        print('Dữ liệu đã được gửi thành công!')
    else:
        print('Có lỗi xảy ra khi gửi dữ liệu.')
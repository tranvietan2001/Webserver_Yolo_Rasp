# import cv2
# import time

# # cap = cv2.VideoCapture(0)
# # video = "/home/antv/Desktop/CodeCV/Project_Robot_Tracking/img/test.mp4"
# cap = cv2.VideoCapture(0)
# # tracker = cv2.legacy.TrackerMOSSE.create()
# tracker = cv2.legacy.TrackerMedianFlow.create()
# # TrackerMedianFlow
# ret, frame = cap.read()
# # lay toa do keo tha chuot
# # bbox = cv2.selectROI("track", frame, False)
# # print(bbox)
# # ----------------
# # toa do tuy chinh\
# # vide_test.mp4
# # x = 810
# # y = 359
# # w = 33
# # h = 22
# # bbox = (x,y,w,h)
# # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) 
# # ----------------
# # test.mp4
# x = 220
# y = 70
# w = 200
# h = 340
# bbox = (x,y,w,h)
# cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) 

# print("Seleck bbox: ",bbox)
# tracker.init(frame, bbox)

# start_time = time.time()
# text_time = ""

# while True:
#     ret,frame = cap.read()

#     frame = cv2.rectangle(frame, (220,70), (420, 410), (0,255,255), 2)

#     if not ret:
#         break
    
#     current_time = time.time()
#     elapsed_time = current_time - start_time
#     text_time = str(int(elapsed_time))

#     if int(elapsed_time) >= 10:  # Dừng vòng lặp sau 10 giây
#         # start_time = time.time()
#         # print("ok")
#         elapsed_time = 10
#         print(int(elapsed_time))
#         text_time = "TRACKING"
    

#         success, bbox = tracker.update(frame)
#         if success:
#             # print(bbox)
#             x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
#             frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)
#             print(x,y)
#         else:
#             print("object lost")
#     cv2.imshow("Window Camera", frame)

#     frame = cv2.putText(frame, text_time, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 1)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# cap.release()
# cv2.destroyAllWindows()



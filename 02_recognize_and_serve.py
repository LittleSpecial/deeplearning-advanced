import cv2
import face_recognition
import pickle
import numpy as np
from pc_recognition_client import send_command_to_robot
from ultralytics import YOLO

DATABASE_FILE = "face_database.pkl"
CONFIDENCE_THRESHOLD = 0.6 # 人脸识别的置信度阈值，越低越容易识别到人脸，但可能会误识别


print("[INFO] Loading the face database...")
try:
    with open(DATABASE_FILE, 'rb') as f:
        database = pickle.load(f)
    print("[INFO] Successfully loaded the face database!")
    # 将数据库中的名字和嵌入向量分开存储，便于快速比对
    known_names = list(database.keys())
    known_encodings = [data["embedding"] for data in database.values()]
except FileNotFoundError:
    print(f"[ERROR] Could not find {DATABASE_FILE}. Please run 01_enroll_faces.py first")
    exit()

print("[INFO] Start the camera...")
# cap = cv2.VideoCapture(0)
PI_STREAM_URL = "http://192.168.43.14:5000/video_feed" 
print(f"正在连接到视频流: {PI_STREAM_URL}")
cap = cv2.VideoCapture(PI_STREAM_URL)

last_identified_name = ""
last_print_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 将BGR图像转换为RGB，因为face_recognition库使用RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 检测画面中的所有人脸
    face_locations = face_recognition.face_locations(rgb_frame)
    # 为检测到的所有人脸计算嵌入向量
    live_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    current_person_identified = False
    # 遍历每个实时检测到的人脸
    for (top, right, bottom, left), live_encoding in zip(face_locations, live_encodings):
        
        # 将实时嵌入与数据库中的所有嵌入进行比对
        matches = face_recognition.compare_faces(known_encodings, live_encoding, tolerance=CONFIDENCE_THRESHOLD)
        # 计算距离，找到最匹配的
        face_distances = face_recognition.face_distance(known_encodings, live_encoding)
        
        name = "Unknown" # 默认为陌生人
        preference = "No Drink" # 默认饮料

        if True in matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                preference = database[name]["preference"]
                current_person_identified = True

        # 绘制人脸框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # 准备显示文本
        display_text = f"{name} wants {preference}"
        # 绘制文本背景框
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        # 绘制文本
        cv2.putText(frame, display_text, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # --- 这里是触发机器人动作的逻辑 ---
        if name != "Unknown":
            # 1. 查询数据库，获取饮料偏好
            preference = database[name]["preference"] # 例如 "可乐"
            # 2. 构建指令字符串
            command = f"FETCH:{preference}" # 例如 "FETCH:Coke"
            # 3. 发送指令给机器人
            send_command_to_robot(command) # 这是一个我们需要实现的通信函数
    
            # (可以在屏幕上显示状态)
            display_text = f"Delivering {preference} to {name}..."
        else:
            command = "STATUS:Idle"
            send_command_to_robot(command)

        if name != "Unknown" and name != last_identified_name:
            print(f"[INFO] Find {name}, favours: {preference}")
            last_identified_name = name

            print("Finding {preference} for {name}...")
            drink_model = YOLO("yolo_weights/best.pt")  # 替换为您的YOLO模型路径
            drink_results = drink_model(frame, stream=True)

            target_found = False
            for result in drink_results:
                for box in result.boxes:
                    detected_class = drink_model.names[int(box.cls[0])]

                    if detected_class.lower() == preference.lower():
                        target_found = True
                        print(f"[INFO] Found {preference} in the frame!")
                        
                        xyxy = box.xyxy[0]  # 获取检测框的坐标
                        screen_width = frame.shape[1]
                        screen_center_x = screen_width // 2
                        x1, _, x2, _ = [int(coord) for coord in xyxy]
                        center_x = (x1 + x2) // 2
                        
                        tolerance = 30

                        robot_coords = convert_pixel_to_robot_coords(center_x, center_y)
                        
                        if robot_coords:
                            command = f"PICK_AT:{robot_coords['x']},{robot_coords['y']},{robot_coords['z']}"
                            print(f"[发送指令] -> {command}")
                            # send_command_to_robot(command)
                        target_found = True
                        break
                        
        if current_person_identified:
            break
        
    if not current_person_identified and last_identified_name:
        # 如果没有识别到人脸，但之前有识别过，清除状态
        print("[INFO] Could not find face, resetting last identified name.")
        last_identified_name = ""
        
    # 显示最终画面
    cv2.imshow("Real-time Face Recognition Vending Machine", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] exit the program.")
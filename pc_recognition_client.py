# -----------------------------------------------------------------------------
# pc_recognition_client.py
# 作用：在PC上运行，接收树莓派视频流，进行AI识别，并将控制指令发回树莓派。
# -----------------------------------------------------------------------------
import cv2
import face_recognition
import pickle
import numpy as np
import requests
import time
from ultralytics import YOLO

# --- 1. 配置 ---
PI_STREAM_URL = "http://192.168.43.14:5000/video_feed" 
PI_COMMAND_URL = "http://192.168.43.14:5000/command"
DATABASE_FILE = "face_database.pkl"
YOLO_MODEL_PATH = "yolo_weights/best.pt"
FACE_CONFIDENCE_THRESHOLD = 0.6
DRINK_CENTERING_TOLERANCE = 30  # 像素容忍度

# --- 2. 加载所有模型 ---
print("[PC INFO] Loading face database...")
try:
    with open(DATABASE_FILE, 'rb') as f:
        database = pickle.load(f)
    known_names = list(database.keys())
    known_encodings = [data["embedding"] for data in database.values()]
    print("[PC INFO] SUCCESS: Face database loaded.")
except FileNotFoundError:
    print(f"[PC ERROR] Could not find {DATABASE_FILE}. Run 01_enroll_faces.py first.")
    exit()

print("[PC INFO] Loading beverage detection model...")
try:
    drink_model = YOLO(YOLO_MODEL_PATH)
    print(f"[PC INFO] SUCCESS: Beverage detection model loaded from {YOLO_MODEL_PATH}.")
except Exception as e:
    print(f"[PC ERROR] Failed to load YOLO model at {YOLO_MODEL_PATH}: {e}")
    exit()

# --- 3. 通信函数 ---
def send_command_to_robot(command, force_send=False):
    """向机器人发送指令，并增加逻辑避免指令刷屏"""
    global last_command # 使用全局变量来跟踪上一个指令
    if command != last_command or force_send:
        print(f"[PC CMD] Sending command: '{command}'")
        try:
            requests.post(PI_COMMAND_URL, json={"command": command}, timeout=1)
            last_command = command
        except requests.exceptions.RequestException:
            print(f"[PC WARN] Failed to send command, network error.")

# --- 4. 主程序 ---
print(f"Connecting to video stream: {PI_STREAM_URL}...")
cap = cv2.VideoCapture(PI_STREAM_URL)

if not cap.isOpened():
    print(f"PC ERROR: Could not connect to video stream {PI_STREAM_URL}")
else:
    print("Connection successful! Starting system...")

    # --- 状态机初始化 ---
    STATE = "SEARCHING_PERSON"
    target_person_name = ""
    target_drink_name = ""
    last_command = ""

    action_timer_start = 0

    APPROACH_BOX_HEIGHT_TARGET = 250 # 当饮料的包围盒高度达到这个像素值时，认为已经足够近
    RETURN_DURATION = 3              # 秒，机器人后退返回的时间
    TURN_180_DURATION = 2            # 秒，机器人原地180度转身需要的时间


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended.")
            break

        # --- 状态一：寻找已注册用户 ---
        if STATE == "SEARCHING_PERSON":
            send_command_to_robot("STATUS:Idle") # 确保小车在找人时是静止的
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
            live_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for live_encoding in live_encodings:
                matches = face_recognition.compare_faces(known_encodings, live_encoding, tolerance=FACE_CONFIDENCE_THRESHOLD)
                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(known_encodings, live_encoding))
                    target_person_name = known_names[best_match_index]
                    target_drink_name = database[target_person_name]["preference"]
                    
                    print(f"\n[STATE CHANGE] Found {target_person_name}, who wants {target_drink_name}.")
                    STATE = "SEARCHING_DRINK"
                    break
                
        elif STATE == "ROTATING_TO_FIND_DRINK":
            send_command_to_robot("TURN:LEFT") # 持续发送左转指令
            drink_results = drink_model(frame, verbose=False)
            target_found = False
            for r in drink_results:
                for box in r.boxes:
                    detected_class = drink_model.names[int(box.cls[0])]
                    if detected_class.lower() == target_drink_name.lower():
                        print(f"\n[STATE CHANGE] Found {target_drink_name}! Now stopping rotation and preparing to approach.")
                        send_command_to_robot("STATUS:Idle", force_send=True) # 立即停止旋转
                        time.sleep(0.5) # 等待机器人稳定
                        STATE = "APPROACHING_DRINK" # 切换到前进状态
                        target_found = True
                        break
                if target_found: break


        elif STATE == "APPROACHING_DRINK":
            send_command_to_robot("MOVE:FORWARD") # 持续发送前进指令
            drink_results = drink_model(frame, verbose=False)
            target_in_sight = False
            for r in drink_results:
                for box in r.boxes:
                    detected_class = drink_model.names[int(box.cls[0])]
                    if detected_class.lower() == target_drink_name.lower():
                        xyxy = box.xyxy[0]
                        box_height = xyxy[3] - xyxy[1]
                        
                        # 检查是否足够近
                        if box_height >= APPROACH_BOX_HEIGHT_TARGET:
                            print(f"\n[STATE CHANGE] Reached {target_drink_name}. Preparing to fetch.")
                            send_command_to_robot(f"FETCH:{target_drink_name}", force_send=True)
                            action_timer_start = time.time() # 启动抓取计时器
                            STATE = "FETCHING_DRINK"
                        
                        target_in_sight = True
                        break
                if target_in_sight: break
            
            if not target_in_sight: # 如果前进时丢失目标，则退回旋转寻找状态
                print("[WARN] Lost sight of the drink while approaching. Returning to rotation search.")
                STATE = "ROTATING_TO_FIND_DRINK"

        # --- 状态四：执行抓取 (计时器状态) ---
        elif STATE == "FETCHING_DRINK":
            # 这是一个延时状态，等待机器人物理抓取动作完成
            if time.time() - action_timer_start > 5: # 假设抓取需要5秒
                print(f"\n[STATE CHANGE] Fetch complete. Returning to user.")
                action_timer_start = time.time() # 重置计时器用于返回
                STATE = "RETURNING_TO_USER"

        # --- 状态五：后退返回 (计时器状态) ---
        elif STATE == "RETURNING_TO_USER":
            send_command_to_robot("MOVE:BACKWARD")
            if time.time() - action_timer_start > RETURN_DURATION:
                print(f"\n[STATE CHANGE] Returned to start position. Turning to face user.")
                action_timer_start = time.time() # 重置计时器用于转身
                STATE = "TURNING_TO_USER"

        # --- 状态六：转身面向用户 (计时器状态) ---
        elif STATE == "TURNING_TO_USER":
            send_command_to_robot("TURN:LEFT") # 可以是左转或右转
            if time.time() - action_timer_start > TURN_180_DURATION:
                print(f"\n[STATE CHANGE] Task fully complete! Ready for next person.")
                send_command_to_robot("STATUS:Idle", force_send=True)
                STATE = "SEARCHING_PERSON" # 回到初始状态

        # 在每一帧上显示当前状态
        cv2.putText(frame, f"STATE: {STATE}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.imshow("Smart Vending Machine - PC Client", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            send_command_to_robot("STATUS:Idle", force_send=True) # 退出前让机器人停止
            break

cap.release()
cv2.destroyAllWindows()
print("Program exited.")
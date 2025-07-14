# robot_control_panel.py
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import requests
import socketio
import cv2
import time  # 添加time模块
import os    # 添加os模块

from ultralytics import YOLO

import face_recognition
import pickle
import numpy as np

# ---------------------------
# 配置
# ---------------------------
PI_ADDRESS = 'http://192.168.43.14:5000'  # 改成你的树莓派IP
VIDEO_STREAM_URL = PI_ADDRESS + '/video_feed'
COMMAND_URL = PI_ADDRESS + '/command'  # 添加命令发送URL


# YOLO模型配置 - 使用官方预训练模型
print("[INFO] Loading YOLO model...")
try:
    # 首先尝试加载自定义饮料检测模型
    custom_model_path = "yolo_weights/best.pt"
    if os.path.exists(custom_model_path):
        yolo_model = YOLO(custom_model_path)
        print(f"[INFO] Custom beverage detection model loaded from {custom_model_path}")
        model_type = "custom"
    else:
        # 如果自定义模型不存在，使用官方预训练模型
        yolo_model = YOLO('yolov8n.pt')
        print("[INFO] Official YOLOv8n model loaded (general object detection)")
        model_type = "official"
        
    print(f"[INFO] Model can detect: {list(yolo_model.names.values())}")
except Exception as e:
    print(f"[ERROR] Failed to load YOLO model: {e}")
    yolo_model = None
    model_type = "none"

# 自动模式相关配置
FACE_CONFIDENCE_THRESHOLD = 0.6
AUTO_MODE_ENABLED = False
current_state = "IDLE"
recognized_person = None
last_recognition_time = 0
RECOGNITION_COOLDOWN = 3  # 秒，避免重复识别

# 摄像头角度控制配置
camera_position = "middle"  # 当前摄像头位置: "up", "down", "middle"
last_camera_command_time = 0
CAMERA_MOVE_COOLDOWN = 2  # 摄像头移动指令间隔，避免频繁发送
auto_stage = "SEARCHING_PERSON"  # 自动模式阶段: "SEARCHING_PERSON", "SEARCHING_DRINK", "ROTATING_SEARCH", "MOVING_TO_DRINK", "GRABBING", "RETURNING", "SERVING"

# 360度搜索配置
rotation_angle = 0  # 当前旋转角度
ROTATION_STEP = 45  # 每次旋转45度
ROTATION_DELAY = 2  # 每个角度停留2秒检测（减少等待时间）
last_rotation_time = 0
drink_found_position = None  # 发现饮料时的角度
original_position = {"x": 0, "y": 0, "angle": 0}  # 机器人原始位置

# 人脸跟踪稳定性配置
face_tracking_data = {}  # 存储每个人脸的跟踪信息
FACE_TRACKING_FRAMES = 2  # 减少到2帧就显示稳定的框
FACE_DISAPPEAR_FRAMES = 8  # 8帧后移除跟踪（约0.53秒）

DATABASE_PATH = "face_database.pkl"  # 修正路径  
try:
    with open(DATABASE_PATH, "rb") as f:
        database = pickle.load(f)
    known_names = list(database.keys())
    known_encodings = [data["embedding"] for data in database.values()]
    print("[INFO] Face database loaded.")
    print(f"[INFO] Found {len(known_names)} registered faces: {known_names}")
except FileNotFoundError:
    print("[ERROR] Face database not found. Please run 01_enroll_faces.py first.")
    known_names = []
    known_encodings = []
    database = {}

# ---------------------------
# 建立 SocketIO 客户端连接
# ---------------------------
sio = socketio.Client()

try:
    sio.connect(PI_ADDRESS)
    print(f"[SOCKETIO] Connected to {PI_ADDRESS}")
except Exception as e:
    print(f"[SOCKETIO ERROR] Failed to connect: {e}")

@sio.event
def connect():
    print("[SOCKETIO] Connected to server")

@sio.event
def disconnect():
    print("[SOCKETIO] Disconnected from server")

@sio.event
def connect_error(data):
    print(f"[SOCKETIO ERROR] Connection failed: {data}")

# ---------------------------
# 模式切换（发送 POST 请求）
# ---------------------------
def switch_mode(mode):
    global AUTO_MODE_ENABLED
    try:
        # 如果有app.py的模式切换接口，使用它
        requests.post(PI_ADDRESS + '/switch_mode', json={'mode': mode}, timeout=2)
        print(f"[MODE] Switched to {mode}")
        AUTO_MODE_ENABLED = (mode == 'auto')
    except Exception as e:
        print(f"[ERROR] Mode switch failed: {e}")
        # 如果app.py没有模式切换接口，只在本地切换
        AUTO_MODE_ENABLED = (mode == 'auto')
        print(f"[MODE] Local mode switched to {mode}")

def send_robot_command(command):
    """发送命令到机器人"""
    try:
        # 首先尝试发送到pi_robot_server.py的接口
        response = requests.post(COMMAND_URL, json={'command': command}, timeout=1)
        if response.status_code == 200:
            print(f"[CMD] Sent to robot: {command}")
            return True
    except Exception as e:
        print(f"[CMD ERROR] Failed to send command {command}: {e}")
    
    # 如果robot server不可用，尝试通过SocketIO发送
    try:
        if command == "MOVE:FORWARD":
            sio.emit('keydown', 'w')
        elif command == "MOVE:BACKWARD":
            sio.emit('keydown', 's')
        elif command == "TURN:LEFT":
            sio.emit('keydown', 'a')
        elif command == "TURN:RIGHT":
            sio.emit('keydown', 'd')
        elif command == "STATUS:Idle":
            sio.emit('keyup', 'x')
        print(f"[CMD] Sent via SocketIO: {command}")
        return True
    except Exception as e:
        print(f"[CMD ERROR] SocketIO command failed: {e}")
        return False

def control_camera_angle(position):
    """控制摄像头角度 - 只有UP和DOWN两个位置"""
    global camera_position, last_camera_command_time
    current_time = time.time()
    
    # 避免频繁发送摄像头控制指令
    if current_time - last_camera_command_time < CAMERA_MOVE_COOLDOWN:
        return
    
    if camera_position == position:
        return  # 已经在目标位置
    
    print(f"[CAMERA] Moving camera from {camera_position} to {position}")
    
    try:
        if position == "up":
            # 发送向上指令 - 用于人脸识别
            sio.emit('keydown', 'u')
            time.sleep(0.2)  # 稍长一点确保动作完成
            sio.emit('keyup', 'u')
            camera_position = "up"
            print("[CAMERA] Camera moved UP for face recognition")
            
        elif position == "down":
            # 发送向下指令 - 用于饮料识别
            sio.emit('keydown', 'j')
            time.sleep(0.2)  # 稍长一点确保动作完成
            sio.emit('keyup', 'j')
            camera_position = "down"
            print("[CAMERA] Camera moved DOWN for beverage detection")
            
        elif position == "middle":
            # 发送中间位置指令
            sio.emit('keydown', 'n')
            time.sleep(0.2)
            sio.emit('keyup', 'n')
            camera_position = "middle"
            print("[CAMERA] Camera moved to MIDDLE position")
            
        last_camera_command_time = current_time
        
    except Exception as e:
        print(f"[CAMERA ERROR] Failed to move camera to {position}: {e}")

def rotate_robot(direction, duration=1.0):
    """旋转机器人"""
    try:
        print(f"[ROBOT DEBUG] Starting rotation {direction} for {duration}s")
        if direction == "left":
            sio.emit('keydown', 'a')
            time.sleep(duration)
            sio.emit('keyup', 'a')
            print(f"[ROBOT] Rotated LEFT for {duration}s")
        elif direction == "right":
            sio.emit('keydown', 'd')
            time.sleep(duration)
            sio.emit('keyup', 'd')
            print(f"[ROBOT] Rotated RIGHT for {duration}s")
        time.sleep(0.3)  # 减少停顿时间，让旋转更连续
        print(f"[ROBOT DEBUG] Rotation {direction} completed")
    except Exception as e:
        print(f"[ROBOT ERROR] Failed to rotate {direction}: {e}")

def move_robot(direction, duration=1.0):
    """移动机器人"""
    try:
        if direction == "forward":
            sio.emit('keydown', 'w')
            time.sleep(duration)
            sio.emit('keyup', 'w')
            print(f"[ROBOT] Moved FORWARD for {duration}s")
        elif direction == "backward":
            sio.emit('keydown', 's')
            time.sleep(duration)
            sio.emit('keyup', 's')
            print(f"[ROBOT] Moved BACKWARD for {duration}s")
        time.sleep(0.5)  # 停顿让机器人稳定
    except Exception as e:
        print(f"[ROBOT ERROR] Failed to move {direction}: {e}")

def grab_drink():
    """抓取饮料"""
    try:
        sio.emit('keydown', 'g')
        time.sleep(0.2)
        sio.emit('keyup', 'g')
        print("[ROBOT] Grabbing drink...")
        time.sleep(3)  # 等待抓取完成
    except Exception as e:
        print(f"[ROBOT ERROR] Failed to grab drink: {e}")

def serve_drink():
    """递送饮料"""
    try:
        sio.emit('keydown', 'r')
        time.sleep(0.2)
        sio.emit('keyup', 'r')
        print("[ROBOT] Serving drink...")
        time.sleep(3)  # 等待递送完成
    except Exception as e:
        print(f"[ROBOT ERROR] Failed to serve drink: {e}")

def return_to_origin():
    """返回原点面向顾客"""
    global drink_found_position
    try:
        # 后退一段距离
        move_robot("backward", 2.0)
        
        # 如果记录了发现饮料的角度，需要反向旋转回到原点
        if drink_found_position is not None:
            # 计算需要反向旋转的角度
            return_angle = (360 - drink_found_position) % 360
            if return_angle > 180:
                # 左转更短
                rotation_time = (360 - return_angle) * 0.8 / 45  # 每45度0.8秒
                rotate_robot("left", rotation_time)
            else:
                # 右转更短  
                rotation_time = return_angle * 0.8 / 45
                rotate_robot("right", rotation_time)
            
            print(f"[ROBOT] Returned to origin from angle {drink_found_position}°")
        else:
            print("[ROBOT] Returned to origin")
            
    except Exception as e:
        print(f"[ROBOT ERROR] Failed to return to origin: {e}")

def check_for_target_drink(detected_objects, user_preference):
    """检查是否发现目标饮料"""
    if not detected_objects:
        return False
    
    # 饮料类别映射
    drink_keywords = {
        'bottle': ['bottle', 'water', 'soda', 'drink'],
        'cup': ['cup', 'coffee', 'tea', 'mug'],
        'wine glass': ['wine', 'glass', 'alcohol'],
        'bowl': ['bowl', 'soup']
    }
    
    # 检查是否有饮料相关物体
    found_drinks = []
    for obj in detected_objects:
        if obj.lower() in ['bottle', 'cup', 'wine glass', 'bowl']:
            found_drinks.append(obj)
    
    if found_drinks:
        print(f"[DRINK DETECTION] Found drinks: {found_drinks}")
        
        # 如果使用官方模型，要更严格地匹配用户偏好
        if model_type == "official":
            # 精确匹配：必须找到与用户偏好完全匹配的饮料
            for drink in found_drinks:
                for keyword in drink_keywords.get(drink, []):
                    if keyword.lower() in user_preference.lower():
                        print(f"[MATCH] Found exact target drink '{drink}' matching preference '{user_preference}'")
                        return True
            
            # 官方模型下，如果没有精确匹配，继续搜索
            print(f"[NO MATCH] Found drinks {found_drinks} but none match preference '{user_preference}', continuing search...")
            return False
        else:
            # 自定义模型下，检测到任何饮料都认为是目标
            for drink in found_drinks:
                for keyword in drink_keywords.get(drink, []):
                    if keyword.lower() in user_preference.lower():
                        print(f"[MATCH] Found target drink '{drink}' matching preference '{user_preference}'")
                        return True
            
            # 如果没有精确匹配，但有饮料，也返回True（通用饮料检测）
            print(f"[MATCH] Found general drinks, proceeding...")
            return True
    
    return False

def update_face_tracking(face_detections, frame_number):
    """更新人脸跟踪数据，减少闪烁"""
    global face_tracking_data
    
    # 如果没有检测到任何人脸，需要处理现有跟踪数据
    if not face_detections:
        print(f"[FACE TRACKING] No faces detected in frame {frame_number}")
        # 清理长时间消失的人脸
        to_remove = []
        for face_id, tracking_info in face_tracking_data.items():
            # 减少稳定性分数
            tracking_info['stable_frames'] = max(0, tracking_info['stable_frames'] - 1)
            
            # 如果人脸消失时间过长，移除
            if frame_number - tracking_info['last_seen_frame'] > FACE_DISAPPEAR_FRAMES:
                to_remove.append(face_id)
                print(f"[FACE TRACKING] Marking face {face_id} for removal - not seen for {frame_number - tracking_info['last_seen_frame']} frames")
        
        for face_id in to_remove:
            del face_tracking_data[face_id]
            print(f"[FACE TRACKING] Removed disappeared face: {face_id}")
        
        return []
    
    # 标记当前帧检测到的人脸ID
    current_frame_faces = set()
    
    # 计算当前检测到的人脸与已跟踪人脸的距离
    current_faces = []
    
    for detection in face_detections:
        name, box, preference = detection
        left, top, right, bottom = box
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        
        # 寻找最近的已跟踪人脸
        best_match = None
        min_distance = float('inf')
        
        for face_id, tracking_info in face_tracking_data.items():
            if tracking_info['name'] == name:
                last_center = tracking_info['center']
                distance = ((center_x - last_center[0]) ** 2 + (center_y - last_center[1]) ** 2) ** 0.5
                if distance < min_distance and distance < 100:  # 放宽到100像素内认为是同一个人脸
                    min_distance = distance
                    best_match = face_id
        
        if best_match:
            # 更新已存在的人脸跟踪
            face_tracking_data[best_match].update({
                'box': box,
                'center': (center_x, center_y),
                'last_seen_frame': frame_number,
                'stable_frames': min(face_tracking_data[best_match]['stable_frames'] + 2, 20),  # 快速增加稳定性
                'preference': preference
            })
            current_faces.append(best_match)
            current_frame_faces.add(best_match)
        else:
            # 创建新的人脸跟踪
            face_id = f"{name}_{frame_number}"
            face_tracking_data[face_id] = {
                'name': name,
                'box': box,
                'center': (center_x, center_y),
                'last_seen_frame': frame_number,
                'stable_frames': 2,  # 直接给较高的初始稳定性
                'preference': preference
            }
            current_faces.append(face_id)
            current_frame_faces.add(face_id)
    
    # 清理长时间消失的人脸
    to_remove = []
    for face_id, tracking_info in face_tracking_data.items():
        if frame_number - tracking_info['last_seen_frame'] > FACE_DISAPPEAR_FRAMES:
            to_remove.append(face_id)
        elif face_id not in current_frame_faces:
            # 未检测到的人脸减少稳定性
            tracking_info['stable_frames'] = max(0, tracking_info['stable_frames'] - 1)
    
    for face_id in to_remove:
        del face_tracking_data[face_id]
        print(f"[FACE TRACKING] Removed disappeared face: {face_id}")
    
    return current_faces

def get_stable_faces():
    """获取稳定跟踪的人脸（减少闪烁）"""
    stable_faces = []
    
    for face_id, tracking_info in face_tracking_data.items():
        # 只要稳定性分数达到阈值就显示，不考虑时间因素
        if tracking_info['stable_frames'] >= FACE_TRACKING_FRAMES:
            stable_faces.append((tracking_info['name'], tracking_info['box'], tracking_info['preference']))
    
    return stable_faces

# ---------------------------
# 键盘控制（只在手动模式下生效）
# ---------------------------
def on_key_press(event):
    if AUTO_MODE_ENABLED:
        print("[INFO] Keyboard control disabled in AUTO mode")
        return
        
    k = event.char.lower()
    if k in ['w', 'a', 's', 'd']:
        sio.emit('keydown', k)
        # 同时显示按键状态
        if k == 'w':
            status_label.config(text="Status: Moving Forward")
        elif k == 's':
            status_label.config(text="Status: Moving Backward")
        elif k == 'a':
            status_label.config(text="Status: Turning Left")
        elif k == 'd':
            status_label.config(text="Status: Turning Right")
    elif k in ['g', 'r']:
        sio.emit('keydown', k)
        status_label.config(text=f"Status: Gripper {'Open' if k == 'g' else 'Close'}")
    elif event.keysym == 'Up':
        sio.emit('keydown', 'u')
        status_label.config(text="Status: Moving Up")
    elif event.keysym == 'Down':
        sio.emit('keydown', 'j')
        status_label.config(text="Status: Moving Down")

def on_key_release(event):
    if AUTO_MODE_ENABLED:
        return
        
    k = event.char.lower()
    if k in ['w', 'a', 's', 'd']:
        sio.emit('keyup', k)
        status_label.config(text="Status: Manual Mode - Use WASD keys to control")

# ---------------------------
# 视频流更新线程
# ---------------------------
def update_video():
    global current_state, recognized_person, last_recognition_time, auto_stage
    global last_rotation_time, rotation_angle, drink_found_position  # 添加全局变量声明
    cap = cv2.VideoCapture(VIDEO_STREAM_URL)
    
    # 优化视频流设置
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲，获取最新帧
    
    # 帧率控制
    fps_limit = 15  # 限制为15FPS，避免闪烁
    frame_time = 1.0 / fps_limit
    last_frame_time = time.time()
    
    # AI处理频率控制
    ai_process_interval = 3  # 每3帧处理一次AI
    frame_count = 0
    
    while not stop_event.is_set():
        current_time = time.time()
        
        # 帧率控制
        if (current_time - last_frame_time) < frame_time:
            time.sleep(frame_time - (current_time - last_frame_time))
            continue
        
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame, retrying...")
            time.sleep(0.1)
            continue
            
        last_frame_time = current_time
        frame_count += 1
        
        # 只在特定帧执行AI处理
        detected_objects = []
        face_detections = []  # 收集当前帧的人脸检测结果
        
        if frame_count % ai_process_interval == 0:
            # 根据自动模式阶段决定使用哪个AI模型
            if AUTO_MODE_ENABLED:
                if auto_stage in ["SEARCHING_PERSON", "SERVING"]:
                    # 人脸识别阶段：只使用face_recognition
                    # print(f"[AI] Using FACE RECOGNITION model in stage: {auto_stage}")  # 减少日志输出
                    if known_encodings:
                        try:
                            # 确保在人脸识别阶段摄像头是抬起的
                            control_camera_angle("up")
                            
                            # 使用小尺寸图像进行人脸识别，提高速度
                            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
                            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                                # 将坐标放大回原始尺寸
                                top *= 2
                                right *= 2
                                bottom *= 2
                                left *= 2
                                
                                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=FACE_CONFIDENCE_THRESHOLD)
                                name = "Unknown"
                                preference = "No preference"
                                
                                if True in matches:
                                    best_index = np.argmin(face_recognition.face_distance(known_encodings, encoding))
                                    if matches[best_index]:
                                        name = known_names[best_index]
                                        preference = database[name]["preference"]
                                        face_detected = True
                                        
                                        # 自动模式下的人脸识别响应
                                        if name != "Unknown":
                                            if (current_time - last_recognition_time > RECOGNITION_COOLDOWN and 
                                                recognized_person != name):
                                                recognized_person = name
                                                last_recognition_time = current_time
                                                print(f"[AUTO] Recognized {name}, preference: {preference}")
                                                
                                                # 切换到360度旋转搜索饮料阶段
                                                auto_stage = "ROTATING_SEARCH"
                                                rotation_angle = 0
                                                last_rotation_time = current_time  # 初始化旋转时间
                                                control_camera_angle("down")  # 摄像头放下寻找饮料
                                                print(f"[AUTO] Starting 360° rotation search for {preference}")
                                                print(f"[AUTO DEBUG] Initial rotation_angle: {rotation_angle}, last_rotation_time: {last_rotation_time}")
                                                
                                                # 立即开始第一次旋转
                                                print("[AUTO] Starting first rotation...")
                                                rotate_robot("right", 0.8)
                                                rotation_angle += ROTATION_STEP
                                                last_rotation_time = current_time

                                # 收集检测结果
                                face_detections.append((name, (left, top, right, bottom), preference))
                        except Exception as e:
                            print(f"[WARN] Face recognition error: {e}")
                            
                elif auto_stage in ["ROTATING_SEARCH", "MOVING_TO_DRINK"]:
                    # 饮料搜索阶段：只使用YOLO
                    # print(f"[AI] Using YOLO model in stage: {auto_stage}")  # 减少日志输出
                    if yolo_model:
                        try:
                            results = yolo_model(frame, verbose=False)
                            for result in results:
                                if result.boxes is not None:
                                    for box in result.boxes:
                                        cls = int(box.cls[0])
                                        label = yolo_model.names[cls]
                                        conf = box.conf[0]
                                        xyxy = box.xyxy[0].cpu().numpy().astype(int)

                                        # 只显示置信度较高的检测结果
                                        if conf > 0.5:
                                            detected_objects.append(label)
                                            
                                            # 画框
                                            x1, y1, x2, y2 = xyxy
                                            
                                            # 根据物体类型使用不同颜色
                                            if label in ['bottle', 'cup', 'wine glass', 'bowl']:
                                                color = (255, 0, 0)  # 饮料相关物体用蓝色
                                            elif label in ['person']:
                                                color = (0, 255, 0)  # 人用绿色
                                            else:
                                                color = (0, 255, 255)  # 其他物体用黄色
                                            
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                            
                                            # 添加标签背景
                                            label_text = f"{label} {conf:.2f}"
                                            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                                            cv2.putText(frame, label_text, (x1, y1 - 5),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        except Exception as e:
                            print(f"[WARN] YOLO detection error: {e}")
                            
            else:
                # 手动模式：同时使用两个模型（保持原有行为）
                # print(f"[AI] Manual mode - using both models")  # 注释掉避免刷屏
                
                # ============ YOLO 物体检测 ============
                if yolo_model:
                    try:
                        results = yolo_model(frame, verbose=False)
                        for result in results:
                            if result.boxes is not None:
                                for box in result.boxes:
                                    cls = int(box.cls[0])
                                    label = yolo_model.names[cls]
                                    conf = box.conf[0]
                                    xyxy = box.xyxy[0].cpu().numpy().astype(int)

                                    # 只显示置信度较高的检测结果
                                    if conf > 0.5:
                                        detected_objects.append(label)
                                        
                                        # 画框
                                        x1, y1, x2, y2 = xyxy
                                        
                                        # 根据物体类型使用不同颜色
                                        if label in ['bottle', 'cup', 'wine glass', 'bowl']:
                                            color = (255, 0, 0)  # 饮料相关物体用蓝色
                                        elif label in ['person']:
                                            color = (0, 255, 0)  # 人用绿色
                                        else:
                                            color = (0, 255, 255)  # 其他物体用黄色
                                        
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                        
                                        # 添加标签背景
                                        label_text = f"{label} {conf:.2f}"
                                        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                                        cv2.putText(frame, label_text, (x1, y1 - 5),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    except Exception as e:
                        print(f"[WARN] YOLO detection error: {e}")

                # ============ 人脸识别（手动模式）============
                face_detected = False
                
                if known_encodings:
                    try:
                        # 使用小尺寸图像进行人脸识别，提高速度
                        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                            # 将坐标放大回原始尺寸
                            top *= 2
                            right *= 2
                            bottom *= 2
                            left *= 2
                            
                            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=FACE_CONFIDENCE_THRESHOLD)
                            name = "Unknown"
                            preference = "No preference"
                            
                            if True in matches:
                                best_index = np.argmin(face_recognition.face_distance(known_encodings, encoding))
                                if matches[best_index]:
                                    name = known_names[best_index]
                                    preference = database[name]["preference"]
                                    face_detected = True

                            # 收集检测结果
                            face_detections.append((name, (left, top, right, bottom), preference))
                    except Exception as e:
                        print(f"[WARN] Face recognition error: {e}")
            
            # ============ 自动模式状态机处理 ============
            if AUTO_MODE_ENABLED:
                try:
                    if auto_stage == "ROTATING_SEARCH":
                        # 360度旋转搜索饮料
                        print(f"[AUTO DEBUG] ROTATING_SEARCH - Current time: {current_time:.2f}, Last rotation: {last_rotation_time:.2f}, Diff: {current_time - last_rotation_time:.2f}, Delay: {ROTATION_DELAY}")
                        
                        if current_time - last_rotation_time > ROTATION_DELAY:
                            # 检查当前角度是否发现了目标饮料
                            if recognized_person and database.get(recognized_person):
                                user_preference = database[recognized_person]["preference"]
                                print(f"[AUTO DEBUG] Checking for drinks. Detected objects: {detected_objects}")
                                print(f"[AUTO DEBUG] User preference: {user_preference}")
                                
                                if check_for_target_drink(detected_objects, user_preference):
                                    print(f"[AUTO] Found target drink at angle {rotation_angle}°!")
                                    drink_found_position = rotation_angle
                                    auto_stage = "MOVING_TO_DRINK"
                                    print("[AUTO] Moving towards the drink...")
                                    # 开始前进到饮料位置
                                    move_robot("forward", 0.5)
                                else:
                                    # 先旋转到下一个角度，再更新角度值
                                    print(f"[AUTO] No target drink found at {rotation_angle}°. Detected: {detected_objects}")
                                    print(f"[AUTO] Rotating to next position...")
                                    rotate_robot("right", 0.8)  # 45度大约需要0.8秒
                                    
                                    # 更新角度
                                    rotation_angle += ROTATION_STEP
                                    if rotation_angle >= 360:
                                        # 完成一圈搜索但没找到，重置搜索
                                        print("[AUTO] Completed 360° search, no target drink found. Restarting...")
                                        rotation_angle = 0
                                        auto_stage = "SEARCHING_PERSON"
                                        last_rotation_time = current_time  # 重置旋转时间
                                        control_camera_angle("up")
                                    else:
                                        print(f"[AUTO] Now at {rotation_angle}° (Step {rotation_angle//ROTATION_STEP}/8)")
                                    
                                    last_rotation_time = current_time
                            else:
                                # 如果没有识别到人员，重置状态
                                print("[AUTO] No recognized person in ROTATING_SEARCH, resetting...")
                                auto_stage = "SEARCHING_PERSON"
                                control_camera_angle("up")
                                last_rotation_time = current_time
                        else:
                            print(f"[AUTO DEBUG] Waiting... Time remaining: {ROTATION_DELAY - (current_time - last_rotation_time):.2f}s")
                    
                    elif auto_stage == "MOVING_TO_DRINK":
                        # 检查是否接近饮料（这里依赖Arduino的超声波自动抓取）
                        # Arduino会在检测到6cm距离时自动抓取
                        # 我们只需要等待一段时间让Arduino完成抓取
                        if not hasattr(control_camera_angle, 'grab_start_time'):
                            control_camera_angle.grab_start_time = current_time
                        
                        if current_time - control_camera_angle.grab_start_time > 8:  # 等待8秒让Arduino完成抓取
                            print("[AUTO] Drink should be grabbed, returning to customer...")
                            auto_stage = "RETURNING"
                            delattr(control_camera_angle, 'grab_start_time')
                            # 开始返回原点
                            return_to_origin()
                    
                    elif auto_stage == "RETURNING":
                        # 返回到原点面向顾客
                        if not hasattr(control_camera_angle, 'return_start_time'):
                            control_camera_angle.return_start_time = current_time
                            
                        if current_time - control_camera_angle.return_start_time > 5:  # 等待5秒完成返回
                            print("[AUTO] Returned to customer, preparing to serve drink...")
                            auto_stage = "SERVING"
                            delattr(control_camera_angle, 'return_start_time')
                            control_camera_angle("up")  # 摄像头抬起面向顾客
                            
                    elif auto_stage == "SERVING":
                        # 递送饮料给顾客
                        if not hasattr(control_camera_angle, 'serve_start_time'):
                            control_camera_angle.serve_start_time = current_time
                            serve_drink()  # 执行递送动作
                            
                        if current_time - control_camera_angle.serve_start_time > 5:  # 等待5秒完成递送
                            print(f"[AUTO] Service completed for {recognized_person}!")
                            # 重置状态，准备下一次服务
                            auto_stage = "SEARCHING_PERSON"
                            recognized_person = None
                            drink_found_position = None
                            rotation_angle = 0
                            last_rotation_time = current_time  # 重置旋转时间
                            delattr(control_camera_angle, 'serve_start_time')
                            face_tracking_data.clear()
                            
                except Exception as e:
                    print(f"[AUTO ERROR] State machine error: {e}")
                    # 发生错误时重置到搜索人员状态
                    auto_stage = "SEARCHING_PERSON"
                    last_rotation_time = time.time()  # 重置旋转时间
                    control_camera_angle("up")
            
            # 更新人脸跟踪数据并绘制稳定的人脸框
            try:
                # 无论是否有检测结果都要更新跟踪数据
                update_face_tracking(face_detections, frame_count)
            except Exception as e:
                print(f"[WARN] Face tracking error: {e}")
            
            # 如果在自动模式下没有检测到已知人脸，重置识别状态
            if AUTO_MODE_ENABLED and not face_detected and recognized_person:
                if current_time - last_recognition_time > RECOGNITION_COOLDOWN * 2:
                    recognized_person = None
                    auto_stage = "SEARCHING_PERSON"  # 重新开始寻找人脸
                    last_rotation_time = current_time  # 重置旋转时间
                    control_camera_angle("up")  # 摄像头重新抬起
                    # 清空人脸跟踪数据
                    face_tracking_data.clear()
                    print("[AUTO] No known face detected, resetting to person search mode")
        
        # 绘制稳定的人脸框（每帧都显示，减少闪烁）
        try:
            stable_faces = get_stable_faces()
            for name, (left, top, right, bottom), preference in stable_faces:
                # 画人脸框
                color = (0, 255, 255) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # 显示姓名和偏好
                label = f"{name}"
                if name != "Unknown":
                    label += f" ({preference})"
                
                # 文本背景
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (left, top - text_height - 10), (left + text_width, top), color, -1)
                
                # 文本
                cv2.putText(frame, label, (left, top - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            print(f"[WARN] Face drawing error: {e}")

        # 显示模式和状态信息（每帧都显示）
        mode_text = "AUTO" if AUTO_MODE_ENABLED else "MANUAL"
        cv2.putText(frame, f"Mode: {mode_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 显示摄像头位置
        camera_text = f"Camera: {camera_position.upper()}"
        cv2.putText(frame, camera_text, (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示自动模式阶段
        if AUTO_MODE_ENABLED:
            try:
                stage_display = auto_stage.replace('_', ' ').title()
                if auto_stage == "ROTATING_SEARCH":
                    stage_display += f" ({rotation_angle}°)"
                elif auto_stage == "MOVING_TO_DRINK" and drink_found_position:
                    stage_display += f" (Found at {drink_found_position}°)"
                
                stage_text = f"Stage: {stage_display}"
                cv2.putText(frame, stage_text, (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            except Exception as e:
                print(f"[WARN] Auto stage display error: {e}")
                cv2.putText(frame, "Stage: Unknown", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if recognized_person:
            cv2.putText(frame, f"Serving: {recognized_person}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示检测到的物体数量
        if detected_objects:
            unique_objects = list(set(detected_objects))
            objects_text = f"Objects: {', '.join(unique_objects[:3])}"
            if len(unique_objects) > 3:
                objects_text += f" +{len(unique_objects)-3} more"
            cv2.putText(frame, objects_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示模型类型
        model_text = f"Model: {model_type.upper()}"
        cv2.putText(frame, model_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ============ 线程安全的GUI更新 ============
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # 使用after方法在主线程中更新GUI
            def update_gui():
                if not stop_event.is_set():
                    video_label.configure(image=imgtk)
                    video_label.imgtk = imgtk
            
            root.after(0, update_gui)
        except Exception as e:
            if not stop_event.is_set():
                print(f"[WARN] GUI update error: {e}")

    cap.release()
    print("[INFO] Video capture released")


# ---------------------------
# GUI 创建
# ---------------------------
root = tk.Tk()
root.title("Smart Robot Control Panel")
root.geometry("1000x800")  # 增加窗口大小
root.bind("<KeyPress>", on_key_press)
root.bind("<KeyRelease>", on_key_release)
root.focus_set()  # 确保窗口获得焦点以接收键盘事件

# 标题
title_label = ttk.Label(root, text="Smart Robot Control Panel", font=('Arial', 16, 'bold'))
title_label.pack(pady=10)

# 模式选择按钮
mode_var = tk.StringVar(value='manual')
ttk.Label(root, text="Control Mode:", font=('Arial', 14)).pack(pady=10)

mode_frame = ttk.Frame(root)
mode_frame.pack()

def set_manual():
    mode_var.set('manual')
    switch_mode('manual')
    status_label.config(text="Status: Manual Mode - Use WASD keys to control")

def set_auto():
    global auto_stage, rotation_angle, drink_found_position, last_rotation_time
    mode_var.set('auto')
    switch_mode('auto')
    auto_stage = "SEARCHING_PERSON"  # 重置到寻找人脸阶段
    rotation_angle = 0  # 重置旋转角度
    drink_found_position = None  # 重置饮料发现位置
    last_rotation_time = time.time()  # 初始化旋转时间为当前时间
    control_camera_angle("up")  # 摄像头抬起准备人脸识别
    print(f"[AUTO DEBUG] Auto mode initialized - auto_stage: {auto_stage}, rotation_angle: {rotation_angle}, last_rotation_time: {last_rotation_time}")
    status_label.config(text="Status: Auto Mode - Searching for person (Camera UP)")

ttk.Button(mode_frame, text="Manual Mode", command=set_manual, width=20).grid(row=0, column=0, padx=10)
ttk.Button(mode_frame, text="Auto Mode", command=set_auto, width=20).grid(row=0, column=1, padx=10)

# 手动控制按钮
control_frame = ttk.LabelFrame(root, text="Manual Controls", padding=10)
control_frame.pack(pady=10)

def send_manual_command(cmd):
    if not AUTO_MODE_ENABLED:
        send_robot_command(cmd)

ttk.Button(control_frame, text="↑ Forward", command=lambda: send_manual_command("MOVE:FORWARD")).grid(row=0, column=1, padx=5, pady=5)
ttk.Button(control_frame, text="← Left", command=lambda: send_manual_command("TURN:LEFT")).grid(row=1, column=0, padx=5, pady=5)
ttk.Button(control_frame, text="Stop", command=lambda: send_manual_command("STATUS:Idle")).grid(row=1, column=1, padx=5, pady=5)
ttk.Button(control_frame, text="→ Right", command=lambda: send_manual_command("TURN:RIGHT")).grid(row=1, column=2, padx=5, pady=5)
ttk.Button(control_frame, text="↓ Backward", command=lambda: send_manual_command("MOVE:BACKWARD")).grid(row=2, column=1, padx=5, pady=5)

# 摄像头控制按钮
def manual_camera_up():
    if not AUTO_MODE_ENABLED:
        control_camera_angle("up")
        status_label.config(text="Status: Camera moved UP")

def manual_camera_down():
    if not AUTO_MODE_ENABLED:
        control_camera_angle("down")
        status_label.config(text="Status: Camera moved DOWN")

ttk.Button(control_frame, text="📹 Camera UP", command=manual_camera_up).grid(row=0, column=0, padx=5, pady=5)
ttk.Button(control_frame, text="📹 Camera DOWN", command=manual_camera_down).grid(row=2, column=0, padx=5, pady=5)

# 测试旋转按钮
def test_rotate_left():
    if not AUTO_MODE_ENABLED:
        rotate_robot("left", 0.8)
        status_label.config(text="Status: Test rotation LEFT")

def test_rotate_right():
    if not AUTO_MODE_ENABLED:
        rotate_robot("right", 0.8)
        status_label.config(text="Status: Test rotation RIGHT")

ttk.Button(control_frame, text="🔄 Test Left", command=test_rotate_left).grid(row=0, column=3, padx=5, pady=5)
ttk.Button(control_frame, text="🔄 Test Right", command=test_rotate_right).grid(row=1, column=3, padx=5, pady=5)

# 测试360度搜索按钮
def test_full_rotation():
    """测试完整的360度旋转搜索"""
    global AUTO_MODE_ENABLED, auto_stage, rotation_angle, last_rotation_time, recognized_person, database
    
    if not AUTO_MODE_ENABLED:
        print("[TEST] Starting 360° rotation test...")
        
        # 创建测试用户数据
        if "Test User" not in database:
            database["Test User"] = {"preference": "bottle"}  # 测试寻找瓶子
        
        auto_stage = "ROTATING_SEARCH"
        rotation_angle = 0
        last_rotation_time = time.time()
        recognized_person = "Test User"  # 模拟识别到用户
        control_camera_angle("down")  # 摄像头放下
        
        # 暂时切换到自动模式进行测试
        AUTO_MODE_ENABLED = True
        status_label.config(text="Status: Testing 360° rotation for bottles...")
        
        # 立即开始第一次旋转
        print("[TEST] Starting first test rotation...")
        rotate_robot("right", 5)
        rotation_angle += ROTATION_STEP
        last_rotation_time = time.time()
        
        # 25秒后恢复手动模式
        def restore_manual():
            global AUTO_MODE_ENABLED, auto_stage
            AUTO_MODE_ENABLED = False
            auto_stage = "SEARCHING_PERSON"
            control_camera_angle("up")  # 摄像头抬起
            status_label.config(text="Status: Manual Mode - Test completed")
        root.after(25000, restore_manual)  # 25秒后恢复手动模式

ttk.Button(control_frame, text="🔄 Test 360°", command=test_full_rotation).grid(row=2, column=3, padx=5, pady=5)

# 模型控制按钮
def reload_yolo_model():
    global yolo_model, model_type
    try:
        custom_model_path = "yolo_weights/best.pt"
        if os.path.exists(custom_model_path):
            yolo_model = YOLO(custom_model_path)
            model_type = "custom"
            model_info = "YOLO Model: Custom (Beverage Detection)"
            print("[INFO] Switched to custom beverage detection model")
        else:
            yolo_model = YOLO('yolov8n.pt')
            model_type = "official"
            model_info = "YOLO Model: Official (General Object Detection)"
            print("[INFO] Using official YOLOv8n model")
        
        model_label.config(text=model_info)
        status_label.config(text=f"Status: Model reloaded - {model_type}")
    except Exception as e:
        print(f"[ERROR] Failed to reload model: {e}")
        status_label.config(text="Status: Model reload failed")

ttk.Button(control_frame, text="Reload Model", command=reload_yolo_model).grid(row=3, column=1, padx=5, pady=5)

# 状态显示
status_label = ttk.Label(root, text="Status: Manual Mode - Use WASD keys to control", font=('Arial', 10))
status_label.pack(pady=5)

# 说明文字
instructions = ttk.Label(root, text="Keyboard: W=Forward, S=Backward, A=Left, D=Right, G/R=Gripper, U/J=Up/Down", 
                        font=('Arial', 9), foreground='gray')
instructions.pack(pady=5)

# 模型信息显示
if yolo_model:
    model_info = f"YOLO Model: {model_type.title()} ({'Custom Beverage Detection' if model_type == 'custom' else 'General Object Detection'})"
else:
    model_info = "YOLO Model: Not loaded"
    
model_label = ttk.Label(root, text=model_info, font=('Arial', 9), foreground='blue')
model_label.pack(pady=5)

# 视频显示区域
video_label = tk.Label(root)
video_label.pack(pady=20)

# 停止线程的事件标志
stop_event = threading.Event()
video_thread = threading.Thread(target=update_video, daemon=True)
video_thread.start()

# 退出清理
def on_close():
    stop_event.set()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()

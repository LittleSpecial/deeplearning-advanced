# app.py
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO
import serial
import io
import time
from picamera2 import Picamera2
import cv2

# ----------------------
# Flask + SocketIO 初始化
# ----------------------
app = Flask(__name__)
socketio = SocketIO(app)

# ----------------------
# 串口初始化
# ----------------------
ser = serial.Serial('/dev/serial0', 115200)

# ----------------------
# 摄像头初始化
# ----------------------
print("[PI] Initializing camera with optimized settings...")
picam2 = Picamera2()
# 配置优化参数
video_config = picam2.create_video_configuration(
    main={
        "size": (640, 480),  # 保留 640x480 以支持 YOLOv7
        "format": "RGB888"   # 切换到 RGB888 确保颜色正确
    },
    controls={
        "FrameDurationLimits": (33333, 33333),  # 固定 30 FPS
        "AeEnable": True,                       # 启用自动曝光以改善亮度
        "AwbEnable": True,                      # 启用自动白平衡以优化颜色
        "AwbMode": 0,                           # 自动白平衡模式
        "AnalogueGain": 2.0,                    # 保留增益以增强亮度
        "Brightness": 0.1,                      # 降低亮度微调以避免过曝
        "Saturation": 1.2,                      # 降低饱和度以更自然
        "Contrast": 1.1                         # 降低对比度以更平滑
    }
)
picam2.configure(video_config)
picam2.start()
print("[PI] Camera started with optimized settings.")

# ----------------------
# 控制模式（manual or auto）
# ----------------------
control_mode = "manual"

@socketio.on('switch_mode')
def switch_mode(data):
    global control_mode
    new_mode = data.get('mode')
    if new_mode in ['manual', 'auto']:
        control_mode = new_mode
        print(f"[MODE] Switched to {control_mode.upper()} mode.")
    else:
        print("[WARN] Invalid mode:", new_mode)

# ----------------------
# 手动控制指令
# ----------------------
@socketio.on('keydown')
def handle_keydown(data):
    global control_mode
    if control_mode != 'manual':
        print("[INFO] Ignoring keydown in AUTO mode.")
        return
    print(f"[KEY DOWN] {data}")
    ser.write(data.encode())

@socketio.on('keyup')
def handle_keyup(data):
    global control_mode
    if control_mode != 'manual':
        print("[INFO] Ignoring keyup in AUTO mode.")
        return
    print(f"[KEY UP] {data}")
    ser.write(b'x')  # 松开统一停止

# ----------------------
# 自动控制指令
# ----------------------
@app.route('/switch_mode', methods=['POST'])
def switch_mode():
    global control_mode
    data = request.get_json()
    if data and 'mode' in data:
        new_mode = data['mode']
        if new_mode in ['manual', 'auto']:
            control_mode = new_mode
            print(f"[MODE] Switched to {control_mode.upper()} mode via HTTP.")
            return {"status": "success", "mode": control_mode}, 200
        else:
            return {"status": "error", "message": "Invalid mode"}, 400
    else:
        return {"status": "error", "message": "Missing mode parameter"}, 400


# ----------------------
# 摄像头视频流路由
# ----------------------
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            # 获取原始帧
            frame = picam2.capture_array("main")
            # 软件翻转（水平和垂直）
            frame = cv2.flip(frame, -1)  # -1 表示水平和垂直翻转
            # 编码为 JPEG，优化质量
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() + b'\r\n')
            # 精确控制帧率（30 FPS）
            time.sleep(0.033)  # 1/30 秒
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------------
# 主程序入口
# ----------------------
if __name__ == '__main__':
    print("[SERVER] Flask-SocketIO server started.")
    socketio.run(app, host='0.0.0.0', port=5000)

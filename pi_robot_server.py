# -----------------------------------------------------------------------------
# pi_robot_server.py
# 作用：在树莓派上运行，提供视频流服务并接收PC指令以控制机器人。
# -----------------------------------------------------------------------------
from flask import Flask, Response, request
from picamera2 import Picamera2
import io
import time
import serial

# --- 0. Arduino串口通信初始化 ---
try:
    arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # 可能需要调整端口号
    time.sleep(2)  # 等待Arduino初始化
    print("[Pi INFO] Arduino connected successfully.")
except Exception as e:
    print(f"[Pi ERROR] Failed to connect to Arduino: {e}")
    arduino = None

def send_arduino_command(command):
    """向Arduino发送指令"""
    if arduino and arduino.is_open:
        try:
            arduino.write((command + '\n').encode())
            arduino.flush()
            print(f"[ARDUINO CMD] Sent: {command}")
            # 等待Arduino响应（可选）
            time.sleep(0.1)
        except Exception as e:
            print(f"[ARDUINO ERROR] Failed to send command: {e}")
    else:
        print("[ARDUINO ERROR] Arduino not connected.")

# --- 1. 摄像头初始化 ---
print("[Pi INFO] Initializing camera...")
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
print("[Pi INFO] Camera started successfully.")

# --- 2. Flask 应用初始化 ---
app = Flask(__name__)

# --- 3. 机器人硬件控制函数 (*** 这是您需要填充具体逻辑的地方 ***) ---
# 建议：将所有与硬件（舵机、马达）直接交互的代码都封装在这些函数里。

def move_forward():
    """控制小车前进"""
    print("[ROBOT ACTION] Moving forward...")
    send_arduino_command("FORWARD")

def move_backward():
    """控制小车后退"""
    print("[ROBOT ACTION] Moving backward...")
    send_arduino_command("BACKWARD")

def turn_left():
    """控制小车左转"""
    print("[ROBOT ACTION] Turning left...")
    send_arduino_command("LEFT")

def turn_right():
    """控制小车右转"""
    print("[ROBOT ACTION] Turning right...")
    send_arduino_command("RIGHT")

def stop_all_motors():
    """停止所有移动"""
    print("[ROBOT ACTION] Stopping all movement.")
    send_arduino_command("STOP")

def fetch_drink(drink_name):
    """执行抓取饮料的机械臂动作序列"""
    print(f"[ROBOT ACTION] Starting pickup for {drink_name}...")
    # 可以发送特定的抓取指令
    send_arduino_command(f"FETCH:{drink_name}")

def return_to_start_position():
    """返回到初始位置"""
    print("[ROBOT ACTION] Returning to start position...")
    send_arduino_command("RETURN_HOME")

def turn_180_degrees():
    """转身180度"""
    print("[ROBOT ACTION] Turning 180 degrees...")
    send_arduino_command("TURN_180")
    
# --- 4. 指令解析与执行 ---
def execute_robot_command(command):
    """解析来自PC的指令并调用相应的机器人动作函数"""
    print(f"[COMMAND RECEIVED] {command}")
    
    if command == "TURN:LEFT":
        turn_left()
    elif command == "TURN:RIGHT":
        turn_right()
    elif command == "MOVE:FORWARD":
        move_forward()
    elif command == "MOVE:BACKWARD":
        move_backward()
    elif command == "RETURN:HOME":
        return_to_start_position() # 调用返回原位的函数
    elif command == "FACE:CUSTOMER":
        turn_180_degrees() # 调用转身函数
    elif command.startswith("FETCH:"):
        drink = command.split(":")[1]
        fetch_drink(drink)
    elif command == "STATUS:Idle":
        stop_all_motors()
    else:
        print(f"[WARN] Unknown command '{command}', doing nothing.")

# --- 5. Flask Web服务 ---
@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    def generate_frames():
        while True:
            stream = io.BytesIO()
            picam2.capture_file(stream, format='jpeg')
            stream.seek(0)
            frame = stream.read()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.01)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/command', methods=['POST'])
def command_receiver():
    """接收PC指令的路由"""
    data = request.get_json()
    if data and 'command' in data:
        cmd = data['command']
        execute_robot_command(cmd)
        return {"status": "success", "command_received": cmd}, 200
    else:
        return {"status": "error", "message": "Invalid command format"}, 400

# --- 6. 主程序入口 ---
if __name__ == '__main__':
    print("[SERVER START] Raspberry Pi Robot Server is running...")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n[SERVER STOP] Shutting down server...")
    finally:
        # 关闭Arduino串口连接
        if arduino and arduino.is_open:
            arduino.close()
            print("[Pi INFO] Arduino connection closed.")
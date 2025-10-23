import cv2
import numpy as np
import pyautogui
import json
import time
import socket
from datetime import datetime

# -----------------------------
# UDP NETWORK CONFIG
# -----------------------------
BROADCAST_PORT = 5006   # PC will broadcast on this port
TRACKING_PORT = 5005    # Pi will send tracking data here
DISCOVERY_TIMEOUT = 5.0 # seconds to wait before re-listening

# UDP sockets
broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
broadcast_sock.bind(("", BROADCAST_PORT))
broadcast_sock.settimeout(DISCOVERY_TIMEOUT)

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_sock.setblocking(False)

pc_ip = None

# -----------------------------
# TRACKING FUNCTIONS
# -----------------------------
def process_frame(frame, buffer, delay, alpha):
    inverted = 255 - frame
    buffer.append(inverted)
    if len(buffer) > delay:
        delayed_inverted = buffer[-(delay + 1)]
        blended = cv2.addWeighted(frame, 1 - alpha, delayed_inverted, alpha, 0)
    else:
        blended = frame
    return blended

def detect_pupil(motion_frame):
    gray = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        return int(x), int(y)
    return None

def write_latest_json(x, y):
    data = {"timestamp": time.time(), "x": x, "y": y}
    with open("outputs.json", "w") as f:
        json.dump(data, f, indent=2)

def append_log(x, y):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("logs.txt", "a") as f:
        f.write(f"[{timestamp}] x={x}, y={y}\n")

def send_udp_data(x, y):
    if not pc_ip:
        return
    payload = {"timestamp": time.time(), "x": x, "y": y}
    try:
        send_sock.sendto(json.dumps(payload).encode(), (pc_ip, TRACKING_PORT))
    except Exception:
        pass

# -----------------------------
# DISCOVERY FUNCTION
# -----------------------------
def discover_pc():
    """Listen for PC broadcast messages and return its IP."""
    print("🔍 Waiting for PC broadcast...")
    while True:
        try:
            data, addr = broadcast_sock.recvfrom(1024)
            if data.decode().strip() == "PC_AVAILABLE":
                print(f"✅ Found PC at {addr[0]}")
                return addr[0]
        except socket.timeout:
            print("⌛ No broadcast received, still listening...")
        except Exception as e:
            print("Network error:", e)
            time.sleep(1)

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    global pc_ip

    test_mode = True
    input_path = "1.mkv"    # set to None for webcam
    loop_video = True

    if input_path and input_path.strip():
        cap = cv2.VideoCapture(input_path)
    else:
        cap = cv2.VideoCapture(0)
        loop_video = False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    buffer = []

    cv2.namedWindow("Motion Extract + Pupil")
    cv2.createTrackbar("Delay", "Motion Extract + Pupil", 2, 30, lambda x: None)
    cv2.createTrackbar("Opacity", "Motion Extract + Pupil", 23, 100, lambda x: None)
    cv2.createTrackbar("Speed", "Motion Extract + Pupil", 10, 50, lambda x: None)

    print("Starting pupil tracking... Press ESC to quit.")

    # --- Auto-discover PC ---
    pc_ip = discover_pc()

    while True:
        ret, frame = cap.read()
        if not ret:
            if loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        delay = max(1, cv2.getTrackbarPos("Delay", "Motion Extract + Pupil"))
        alpha = cv2.getTrackbarPos("Opacity", "Motion Extract + Pupil") / 100.0
        speed_slider = cv2.getTrackbarPos("Speed", "Motion Extract + Pupil")
        playback_speed = max(1, speed_slider) / 10.0

        motion_frame = process_frame(frame, buffer, delay, alpha)
        pupil = detect_pupil(motion_frame)

        if pupil:
            x, y = pupil
            cv2.circle(motion_frame, (x, y), 5, (0, 0, 255), -1)
            write_latest_json(x, y)
            append_log(x, y)
            send_udp_data(x, y)

            if not test_mode:
                screen_w, screen_h = pyautogui.size()
                pyautogui.moveTo(x % screen_w, y % screen_h)

        cv2.imshow("Motion Extract + Pupil", motion_frame)

        if cv2.waitKey(int((1000 / fps) / playback_speed)) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    send_sock.close()
    broadcast_sock.close()

if __name__ == "__main__":
    main()

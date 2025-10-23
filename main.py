import cv2
import numpy as np
import pyautogui
import json
import time
import socket
from datetime import datetime

# -----------------------------
# UDP CONFIGURATION
# -----------------------------
PC_IP = "192.168.1.100"   # 🔹 Change this to your PC's IP
PC_PORT = 5005             # 🔹 Same as the receiver’s listening port
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setblocking(False)    # non-blocking (won’t freeze if no network)

# -----------------------------
# EXISTING FUNCTIONS
# -----------------------------
def process_frame(frame, buffer, delay, alpha):
    inverted = 255 - frame
    buffer.append(inverted)
    if len(buffer) > delay:
        delayed_inverted = buffer[-(delay+1)]
        blended = cv2.addWeighted(frame, 1-alpha, delayed_inverted, alpha, 0)
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

def nothing(x):
    pass

def write_latest_json(x, y):
    data = {"timestamp": time.time(), "x": x, "y": y}
    with open("outputs.json", "w") as f:
        json.dump(data, f, indent=2)

def append_log(x, y):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("logs.txt", "a") as f:
        f.write(f"[{timestamp}] x={x}, y={y}\n")

# -----------------------------
# NEW: Send data via UDP
# -----------------------------
def send_udp_data(x, y):
    payload = {"timestamp": time.time(), "x": x, "y": y}
    message = json.dumps(payload).encode()
    try:
        sock.sendto(message, (PC_IP, PC_PORT))
    except Exception:
        # Ignore transient network errors
        pass

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
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
    cv2.createTrackbar("Delay", "Motion Extract + Pupil", 2, 30, nothing)
    cv2.createTrackbar("Opacity", "Motion Extract + Pupil", 23, 100, nothing)
    cv2.createTrackbar("Speed", "Motion Extract + Pupil", 10, 50, nothing)

    print("Starting pupil tracking... Press ESC to quit.")

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
            send_udp_data(x, y)  # 🔹 SEND OVER NETWORK HERE

            if not test_mode:
                screen_w, screen_h = pyautogui.size()
                pyautogui.moveTo(x % screen_w, y % screen_h)

        cv2.imshow("Motion Extract + Pupil", motion_frame)

        if cv2.waitKey(int((1000 / fps) / playback_speed)) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()

if __name__ == "__main__":
    main()

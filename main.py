import cv2
import numpy as np
import json
import time
import socket

# -----------------------------
# SETTINGS
# -----------------------------
SHOW_WINDOW = False       # 🔹 Set to False for headless operation
INPUT_PATH = "1.mkv"      # 🔹 For testing; set to None for webcam
LOOP_VIDEO = True         # 🔹 Loop playback if using a test file

BROADCAST_PORT = 5006     # PC broadcasts here
TRACKING_PORT = 5005      # Pi sends data here
DISCOVERY_TIMEOUT = 5.0   # Seconds before re-listening

# -----------------------------
# NETWORK INITIALIZATION
# -----------------------------
broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
broadcast_sock.bind(("", BROADCAST_PORT))
broadcast_sock.settimeout(DISCOVERY_TIMEOUT)

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_sock.setblocking(False)

pc_ip = None

# -----------------------------
# IMAGE PROCESSING
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

# -----------------------------
# NETWORKING
# -----------------------------
def discover_pc():
    """Listen for PC broadcasts and return its IP when found."""
    print("🔍 Waiting for PC broadcast...")
    while True:
        try:
            data, addr = broadcast_sock.recvfrom(1024)
            if data.decode().strip() == "PC_AVAILABLE":
                print(f"✅ Found PC at {addr[0]}")
                return addr[0]
        except socket.timeout:
            print("⌛ Still listening for PC...")
        except Exception as e:
            print("Network error:", e)
            time.sleep(1)

def send_udp_data(x, y):
    """Send tracking data to the discovered PC."""
    if not pc_ip:
        return
    payload = {"timestamp": time.time(), "x": x, "y": y}
    try:
        send_sock.sendto(json.dumps(payload).encode(), (pc_ip, TRACKING_PORT))
    except Exception:
        pass

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    global pc_ip

    # --- Input setup ---
    if INPUT_PATH and INPUT_PATH.strip():
        cap = cv2.VideoCapture(INPUT_PATH)
    else:
        cap = cv2.VideoCapture(0)
        LOOP_VIDEO = False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    buffer = []

    # --- Only create UI if display is enabled ---
    if SHOW_WINDOW:
        cv2.namedWindow("Motion Extract + Pupil")
        cv2.createTrackbar("Delay", "Motion Extract + Pupil", 2, 30, lambda x: None)
        cv2.createTrackbar("Opacity", "Motion Extract + Pupil", 23, 100, lambda x: None)
        cv2.createTrackbar("Speed", "Motion Extract + Pupil", 10, 50, lambda x: None)
    else:
        # Default static settings for headless use
        delay = 2
        alpha = 0.23
        playback_speed = 1.0

    print("🎥 Starting pupil tracking... Press ESC to quit (if window shown).")

    # --- Auto-discover PC ---
    pc_ip = discover_pc()

    while True:
        ret, frame = cap.read()
        if not ret:
            if LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        if SHOW_WINDOW:
            delay = max(1, cv2.getTrackbarPos("Delay", "Motion Extract + Pupil"))
            alpha = cv2.getTrackbarPos("Opacity", "Motion Extract + Pupil") / 100.0
            speed_slider = cv2.getTrackbarPos("Speed", "Motion Extract + Pupil")
            playback_speed = max(1, speed_slider) / 10.0

        motion_frame = process_frame(frame, buffer, delay, alpha)
        pupil = detect_pupil(motion_frame)

        if pupil:
            x, y = pupil
            send_udp_data(x, y)
            if SHOW_WINDOW:
                cv2.circle(motion_frame, (x, y), 5, (0, 0, 255), -1)

        if SHOW_WINDOW:
            cv2.imshow("Motion Extract + Pupil", motion_frame)
            if cv2.waitKey(int((1000 / fps) / playback_speed)) & 0xFF == 27:
                break
        else:
            time.sleep(1 / fps / playback_speed)

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    send_sock.close()
    broadcast_sock.close()

if __name__ == "__main__":
    main()

import socket
import threading
import json
import time

LISTEN_PORT = 5005
BROADCAST_PORT = 5006

# --- Receiver for tracking data ---
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind(("", LISTEN_PORT))

# --- Broadcaster to announce PC presence ---
broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

def broadcast_thread():
    while True:
        msg = "PC_AVAILABLE"
        broadcast_sock.sendto(msg.encode(), ("<broadcast>", BROADCAST_PORT))
        time.sleep(1)

threading.Thread(target=broadcast_thread, daemon=True).start()

print("✅ PC ready: broadcasting and listening for eye-tracking data...")

while True:
    data, addr = recv_sock.recvfrom(4096)
    try:
        tracking_data = json.loads(data.decode())
        print(f"📍 {addr[0]} -> x={tracking_data['x']}, y={tracking_data['y']}")
    except Exception:
        pass

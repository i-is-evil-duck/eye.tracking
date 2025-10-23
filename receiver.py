import socket
import threading
import json
import time
from datetime import datetime

LISTEN_PORT = 5005
BROADCAST_PORT = 5006

LOG_FILE = "tracking_log.txt"
LATEST_FILE = "latest.json"

# --- Receiver for tracking data ---
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind(("", LISTEN_PORT))

# --- Broadcaster to announce PC presence ---
broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

def broadcast_thread():
    """Continuously broadcast 'PC_AVAILABLE' every second."""
    while True:
        try:
            broadcast_sock.sendto(b"PC_AVAILABLE", ("<broadcast>", BROADCAST_PORT))
        except Exception as e:
            print("Broadcast error:", e)
        time.sleep(1)

threading.Thread(target=broadcast_thread, daemon=True).start()

print("receiver started.")
print("Broadcasting 'PC_AVAILABLE' every second.")
print("Data logging enabled\n")

def log_to_file(data):
    """Append tracking data to a text log and save latest as JSON."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    x, y = data.get("x"), data.get("y")

    # Append to text log
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] x={x}, y={y}\n")

    # Update latest.json
    with open(LATEST_FILE, "w") as f:
        json.dump(data, f, indent=2)

# --- Main loop ---
while True:
    try:
        data, addr = recv_sock.recvfrom(4096)
        tracking_data = json.loads(data.decode())

        # Log received data
        log_to_file(tracking_data)

    except json.JSONDecodeError:
        # Ignore bad packets
        continue
    except KeyboardInterrupt:
        print("\n🛑 Exiting receiver.")
        break
    except Exception as e:
        print("Receiver error:", e)

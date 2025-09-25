import cv2
import numpy as np
import pyautogui

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

def main():
    # --- Settings ---
    test_mode = False        # disables cursor movement if True
    input_path = "test.mov" # set to None for webcam
    loop_video = True

    # --- Input source ---
    if input_path and input_path.strip():
        cap = cv2.VideoCapture(input_path)
    else:
        cap = cv2.VideoCapture(0)
        loop_video = False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    buffer = []
    trail_mode = False
    trail_points = []

    # --- UI Sliders ---
    cv2.namedWindow("Motion Extract + Pupil")
    cv2.createTrackbar("Delay", "Motion Extract + Pupil", 3, 30, nothing)
    cv2.createTrackbar("Opacity", "Motion Extract + Pupil", 50, 100, nothing)
    cv2.createTrackbar("Speed", "Motion Extract + Pupil", 10, 50, nothing)  # 10 = normal

    while True:
        ret, frame = cap.read()
        if not ret:
            if loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # --- Get slider values ---
        delay = max(1, cv2.getTrackbarPos("Delay", "Motion Extract + Pupil"))
        alpha = cv2.getTrackbarPos("Opacity", "Motion Extract + Pupil") / 100.0
        speed_slider = cv2.getTrackbarPos("Speed", "Motion Extract + Pupil")
        playback_speed = max(1, speed_slider) / 10.0

        # --- Process ---
        motion_frame = process_frame(frame, buffer, delay, alpha)
        pupil = detect_pupil(motion_frame)

        if pupil:
            x, y = pupil
            if trail_mode:
                trail_points.append((x, y))
                # draw all previous points in red
                for pt in trail_points[:-1]:
                    cv2.circle(motion_frame, pt, 3, (0, 0, 255), -1)
                # draw current point in green
                cv2.circle(motion_frame, (x, y), 5, (0, 255, 0), -1)
            else:
                cv2.circle(motion_frame, (x, y), 5, (0, 255, 0), -1)

            if not test_mode:
                screen_w, screen_h = pyautogui.size()
                pyautogui.moveTo(x % screen_w, y % screen_h)

        cv2.imshow("Motion Extract + Pupil", motion_frame)

        key = cv2.waitKey(int((1000 / fps) / playback_speed)) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key == ord("t"):  # toggle trail mode
            trail_mode = not trail_mode
            if not trail_mode:
                trail_points.clear()  # reset when turning off

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import cv2
import time
import config

from modules.hand_tracker import HandTracker
from modules.canvas_manager import CanvasManager
from modules.gesture_recognizer import GestureRecognizer
from modules.solver import MathSolver


tracker = HandTracker()
canvas = CanvasManager(config.WIDTH, config.HEIGHT)
gesture = GestureRecognizer()
solver = MathSolver()

cap = cv2.VideoCapture(config.CAMERA_ID)

if not cap.isOpened():
    print("❌ Camera failed to open.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.HEIGHT)

prev_point = None
was_fist = False
was_drawing = False

recognized_latex = ""
recognized_result = ""
show_result = False

last_erase_time = 0
ERASE_COOLDOWN = 0.8

last_solve_time = 0
SOLVE_COOLDOWN = 1.0

print("🎯 Air Math Solver Started!\n")
print("☝️  Index       → Draw")
print("✊  Fist        → Pause / Move (start a new stroke)")
print("✌️  Two Fingers → Erase Last Stroke")
print("'s' → Solve")
print("'c' → Clear Canvas")
print("'q' → Quit\n")


while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = tracker.process(frame)

    if results and results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        index_tip = hand_landmarks.landmark[8]
        x = int(index_tip.x * config.WIDTH)
        y = int(index_tip.y * config.HEIGHT)

        x = max(0, min(config.WIDTH - 1, x))
        y = max(0, min(config.HEIGHT - 1, y))
        current_point = (x, y)

       
        if gesture.is_two_fingers_up(hand_landmarks):
            now = time.time()
            if now - last_erase_time > ERASE_COOLDOWN:
                canvas.erase_last_stroke()
                last_erase_time = now
                print("🗑 Last stroke erased")
            prev_point = None
            was_fist = False
            was_drawing = False

       
        elif gesture.is_drawing_mode(hand_landmarks):

            # Every time we transition INTO drawing, start a clean stroke
            if not was_drawing:
                canvas.start_new_stroke()
                prev_point = None

            if prev_point is not None:
                dist = ((x - prev_point[0]) ** 2 + (y - prev_point[1]) ** 2) ** 0.5
                if dist > config.MOVEMENT_THRESHOLD:
                    canvas.add_point(x, y)
            else:
                canvas.add_point(x, y)

            prev_point = current_point
            was_fist = False
            was_drawing = True

        elif gesture.is_fist(hand_landmarks):

            prev_point = None
            was_drawing = False

            if not was_fist:
                canvas.start_new_stroke()
                print("✊ Stroke saved")

            was_fist = True

        
        else:
            prev_point = None
            was_fist = False
            was_drawing = False

    else:
        # No hand detected
        prev_point = None
        was_fist = False
        was_drawing = False

    overlay = canvas.get_canvas()

    # Blend canvas over the live camera feed
    combined = cv2.addWeighted(
        frame,   1 - config.CANVAS_ALPHA,
        overlay, config.CANVAS_ALPHA, 0
    )

    # HUD text
    cv2.putText(
        combined,
        "Index=Draw | Fist=Pause | Two Fingers=Erase | s=Solve | c=Clear",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Show result overlay — stays visible until canvas is cleared
    if show_result:
        # Semi-transparent dark background behind text so it's readable
        overlay_bg = combined.copy()
        cv2.rectangle(overlay_bg, (20, 90), (config.WIDTH - 20, 210), (0, 0, 0), -1)
        cv2.addWeighted(overlay_bg, 0.5, combined, 0.5, 0, combined)

        cv2.putText(
            combined,
            f"LaTeX:  {recognized_latex}",
            (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )
        cv2.putText(
            combined,
            f"Result = {recognized_result}",
            (40, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    cv2.imshow("Air Math Solver", combined)

    
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("c"):
        canvas.clear()
        recognized_latex = ""
        recognized_result = ""
        show_result = False
        prev_point = None
        was_fist = False
        was_drawing = False
        print("🧹 Canvas Cleared")

    elif key == ord("s"):
        now = time.time()
        if now - last_solve_time > SOLVE_COOLDOWN:
            last_solve_time = now

            # Save any unfinished stroke before solving
            canvas.start_new_stroke()

            canvas_img = canvas.get_canvas()
            gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)

            if cv2.countNonZero(gray) > 200:
                print("🔍 Recognizing equation...")
                try:
                    latex, result = solver.recognize_and_solve(canvas_img)
                    recognized_latex = str(latex)
                    recognized_result = str(result)
                    show_result = True
                    print(f"LaTeX:  {recognized_latex}")
                    print(f"Result: {recognized_result}\n")
                except Exception as e:
                    recognized_latex = "Recognition Failed"
                    recognized_result = str(e)
                    show_result = True
                    print("❌ Solve Error:", e)
            else:
                print("⚠️  Canvas is empty, nothing to solve.")


cap.release()
cv2.destroyAllWindows()
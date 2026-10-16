import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

canvas = None
previous_point = None

draw_color = (0, 255, 0)
color_name = "GREEN"
eraser_mode = False

# Toolbar settings
toolbar_height = 80

buttons = [
    ("RED", (0, 0, 255)),
    ("GREEN", (0, 255, 0)),
    ("BLUE", (255, 0, 0)),
    ("YELLOW", (0, 255, 255)),
    ("WHITE", (255, 255, 255)),
    ("ERASER", None),
    ("CLEAR", None)
]

button_width = 90


def mouse_callback(event, x, y, flags, param):

    global draw_color
    global color_name
    global eraser_mode
    global canvas
    global previous_point

    if event == cv2.EVENT_LBUTTONDOWN:

        if y < toolbar_height:

            button_index = x // button_width

            if button_index < len(buttons):

                button_name, button_color = buttons[button_index]

                if button_name == "ERASER":
                    eraser_mode = True
                    color_name = "ERASER"

                elif button_name == "CLEAR":
                    canvas = np.zeros_like(canvas)
                    previous_point = None

                else:
                    draw_color = button_color
                    color_name = button_name
                    eraser_mode = False


cv2.namedWindow("AirDraw - Stage 9", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AirDraw - Stage 9", 1280, 720)
cv2.setMouseCallback("AirDraw - Stage 9", mouse_callback)

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not access the camera")
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        button_width = frame.shape[1] // len(buttons)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        # Draw toolbar
        for i, (name, color) in enumerate(buttons):

            x1 = i * button_width
            x2 = x1 + button_width

            cv2.rectangle(
                frame,
                (x1, 0),
                (x2, toolbar_height),
                (40, 40, 40),
                -1
            )

            cv2.rectangle(
                frame,
                (x1, 0),
                (x2, toolbar_height),
                (255, 255, 255),
                2
            )

            text_color = color if color is not None else (255, 255, 255)

            cv2.putText(
                frame,
                name,
                (x1 + 15, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2
            )

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            fingertip = hand[8]

            x = int(fingertip.x * frame.shape[1])
            y = int(fingertip.y * frame.shape[0])

            # Finger positions
            index_up = hand[8].y < hand[6].y

            middle_down = hand[12].y > hand[10].y
            ring_down = hand[16].y > hand[14].y
            little_down = hand[20].y > hand[18].y

            middle_up = hand[12].y < hand[10].y
            ring_up = hand[16].y < hand[14].y
            little_up = hand[20].y < hand[18].y

            drawing_gesture = (
                index_up
                and middle_down
                and ring_down
                and little_down
            )

            eraser_gesture = (
                index_up
                and middle_up
                and ring_up
                and little_up
            )

            cv2.circle(
                frame,
                (x, y),
                10,
                (0, 0, 255),
                -1
            )

            if eraser_mode or eraser_gesture:

                cv2.putText(
                    frame,
                    "ERASER",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.circle(
                    canvas,
                    (x, y),
                    30,
                    (0, 0, 0),
                    -1
                )

                previous_point = None

            elif drawing_gesture:

                cv2.putText(
                    frame,
                    "DRAWING - " + color_name,
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    draw_color,
                    2
                )

                if previous_point is not None:

                    cv2.line(
                        canvas,
                        previous_point,
                        (x, y),
                        draw_color,
                        5
                    )

                previous_point = (x, y)

            else:

                cv2.putText(
                    frame,
                    "NOT DRAWING",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                previous_point = None

        else:

            previous_point = None

        combined = cv2.add(frame, canvas)

        cv2.imshow("AirDraw - Stage 9", combined)

        key = cv2.waitKey(1) & 0xFF

        # Keyboard backup controls
        if key == ord('r'):
            draw_color = (0, 0, 255)
            color_name = "RED"
            eraser_mode = False

        elif key == ord('g'):
            draw_color = (0, 255, 0)
            color_name = "GREEN"
            eraser_mode = False

        elif key == ord('b'):
            draw_color = (255, 0, 0)
            color_name = "BLUE"
            eraser_mode = False

        elif key == ord('y'):
            draw_color = (0, 255, 255)
            color_name = "YELLOW"
            eraser_mode = False

        elif key == ord('w'):
            draw_color = (255, 255, 255)
            color_name = "WHITE"
            eraser_mode = False

        elif key == ord('c'):
            canvas = np.zeros_like(frame)
            previous_point = None

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import time

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

# Used for pinky selection
hover_button = -1
hover_start_time = None
selected_button = 1


def select_button(button_index):

    global draw_color
    global color_name
    global eraser_mode
    global canvas
    global previous_point
    global selected_button

    name, color = buttons[button_index]

    if name == "ERASER":

        eraser_mode = True
        color_name = "ERASER"

    elif name == "CLEAR":

        canvas = np.zeros_like(canvas)
        previous_point = None

    else:

        draw_color = color
        color_name = name
        eraser_mode = False

    selected_button = button_index


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

        pinky_x = None
        pinky_y = None

        # --------------------------------------------------
        # HAND DETECTION
        # --------------------------------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Index fingertip
            index_x = int(hand[8].x * frame.shape[1])
            index_y = int(hand[8].y * frame.shape[0])

            # Pinky fingertip
            pinky_x = int(hand[20].x * frame.shape[1])
            pinky_y = int(hand[20].y * frame.shape[0])

            # Finger states
            index_up = hand[8].y < hand[6].y

            middle_down = hand[12].y > hand[10].y
            ring_down = hand[16].y > hand[14].y
            little_down = hand[20].y > hand[18].y

            middle_up = hand[12].y < hand[10].y
            ring_up = hand[16].y < hand[14].y
            little_up = hand[20].y < hand[18].y

            # Normal drawing gesture
            drawing_gesture = (
                index_up
                and middle_down
                and ring_down
                and little_down
            )

            # Eraser gesture
            eraser_gesture = (
                index_up
                and middle_up
                and ring_up
                and little_up
            )

            # Pinky selection gesture
            pinky_selection = (
                little_up
                and not drawing_gesture
                and not eraser_gesture
            )

            # --------------------------------------------------
            # PINKY TOOLBAR CONTROL
            # --------------------------------------------------

            if pinky_selection and pinky_y < toolbar_height:

                button_index = pinky_x // button_width

                if button_index < len(buttons):

                    # Highlight button being pointed at
                    hover_button = button_index

                    if hover_start_time is None:
                        hover_start_time = time.time()

                    # Select after pinky stays there for 0.5 sec
                    if time.time() - hover_start_time > 0.5:

                        select_button(button_index)

                        hover_start_time = None

            else:

                hover_button = -1
                hover_start_time = None

            # --------------------------------------------------
            # SHOW INDEX FINGERTIP
            # --------------------------------------------------

            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (0, 0, 255),
                -1
            )

            # Show pinky fingertip
            cv2.circle(
                frame,
                (pinky_x, pinky_y),
                8,
                (255, 0, 255),
                -1
            )

            # --------------------------------------------------
            # DRAWING / ERASING
            # --------------------------------------------------

            # Don't draw while pinky is selecting a button
            if pinky_selection and pinky_y < toolbar_height:

                cv2.putText(
                    frame,
                    "PINKY - SELECT TOOL",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 255),
                    2
                )

                previous_point = None

            elif eraser_mode or eraser_gesture:

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
                    (index_x, index_y),
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
                        (index_x, index_y),
                        draw_color,
                        5
                    )

                previous_point = (index_x, index_y)

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
            hover_button = -1
            hover_start_time = None

        # --------------------------------------------------
        # DRAW TOOLBAR
        # --------------------------------------------------

        for i, (name, color) in enumerate(buttons):

            x1 = i * button_width
            x2 = x1 + button_width

            # Normal button
            cv2.rectangle(
                frame,
                (x1, 0),
                (x2, toolbar_height),
                (40, 40, 40),
                -1
            )

            # Highlight selected button
            if i == selected_button:

                cv2.rectangle(
                    frame,
                    (x1 + 4, 4),
                    (x2 - 4, toolbar_height - 4),
                    (70, 70, 70),
                    -1
                )

            # Highlight button under pinky
            if i == hover_button:

                cv2.rectangle(
                    frame,
                    (x1 + 2, 2),
                    (x2 - 2, toolbar_height - 2),
                    (255, 255, 255),
                    4
                )

            else:

                cv2.rectangle(
                    frame,
                    (x1, 0),
                    (x2, toolbar_height),
                    (255, 255, 255),
                    2
                )

            if color is not None:
                text_color = color
            else:
                text_color = (255, 255, 255)

            cv2.putText(
                frame,
                name,
                (x1 + 15, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2
            )

        # --------------------------------------------------
        # COMBINE CAMERA + DRAWING
        # --------------------------------------------------

        combined = cv2.add(frame, canvas)

        cv2.imshow(
            "AirDraw - Stage 10",
            combined
        )

        key = cv2.waitKey(1) & 0xFF

        # Keyboard backup
        if key == ord('q'):
            break

        elif key == ord('r'):
            draw_color = (0, 0, 255)
            color_name = "RED"
            eraser_mode = False
            selected_button = 0

        elif key == ord('g'):
            draw_color = (0, 255, 0)
            color_name = "GREEN"
            eraser_mode = False
            selected_button = 1

        elif key == ord('b'):
            draw_color = (255, 0, 0)
            color_name = "BLUE"
            eraser_mode = False
            selected_button = 2

        elif key == ord('y'):
            draw_color = (0, 255, 255)
            color_name = "YELLOW"
            eraser_mode = False
            selected_button = 3

        elif key == ord('w'):
            draw_color = (255, 255, 255)
            color_name = "WHITE"
            eraser_mode = False
            selected_button = 4

        elif key == ord('c'):
            canvas = np.zeros_like(frame)
            previous_point = None
            selected_button = 6

cap.release()
cv2.destroyAllWindows()
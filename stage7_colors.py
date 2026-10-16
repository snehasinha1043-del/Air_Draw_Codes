import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

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

# Starting color
draw_color = (0, 255, 0)
color_name = "GREEN"

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not access the camera")
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

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

            # Drawing gesture
            drawing_mode = (
                index_up
                and middle_down
                and ring_down
                and little_down
            )

            # Eraser gesture
            eraser_mode = (
                index_up
                and middle_up
                and ring_up
                and little_up
            )

            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

            if drawing_mode:

                cv2.putText(
                    frame,
                    "DRAWING",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    draw_color,
                    2
                )

                cv2.putText(
                    frame,
                    "Color: " + color_name,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
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

            elif eraser_mode:

                cv2.putText(
                    frame,
                    "ERASER",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
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

            else:

                cv2.putText(
                    frame,
                    "NOT DRAWING",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

                previous_point = None

        else:
            previous_point = None

        combined = cv2.add(frame, canvas)

        cv2.imshow("AirDraw - Stage 7", combined)

        key = cv2.waitKey(1) & 0xFF

        # Color selection
        if key == ord('r'):
            draw_color = (0, 0, 255)
            color_name = "RED"

        elif key == ord('g'):
            draw_color = (0, 255, 0)
            color_name = "GREEN"

        elif key == ord('b'):
            draw_color = (255, 0, 0)
            color_name = "BLUE"

        elif key == ord('y'):
            draw_color = (0, 255, 255)
            color_name = "YELLOW"

        elif key == ord('w'):
            draw_color = (255, 255, 255)
            color_name = "WHITE"

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
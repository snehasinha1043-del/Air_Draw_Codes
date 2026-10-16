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

            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

            if previous_point is not None:
                cv2.line(
                    canvas,
                    previous_point,
                    (x, y),
                    (0, 255, 0),
                    5
                )

            previous_point = (x, y)

        else:
            previous_point = None

        combined = cv2.add(frame, canvas)

        cv2.imshow("AirDraw - Stage 4", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
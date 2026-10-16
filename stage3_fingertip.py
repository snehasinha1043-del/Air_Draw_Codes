import cv2
import mediapipe as mp

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

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not access the camera")
            break

        frame = cv2.flip(frame, 1)

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

            cv2.circle(frame, (x, y), 12, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"Index: ({x}, {y})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("AirDraw - Fingertip Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
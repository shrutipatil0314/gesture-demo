import cv2
import mediapipe as mp
import numpy as np

# --- Your gesture_dict and classify_gesture() go here ---
# (Use the 15-gesture version we wrote earlier)

gesture_dict = {
    "OPEN_HAND": "Hello",
    "THUMBS_UP": "Yes",
    "THUMBS_DOWN": "No",
    "FIST": "Stop",
    "PEACE": "Peace",
    "OK_SIGN": "Okay",
    "POINT_UP": "Look up",
    "POINT_DOWN": "Look down",
    "POINT_RIGHT": "Go right",
    "POINT_LEFT": "Go left",
    "ROCK_SIGN": "Rock on",
    "CALL_ME": "Call me",
    "PINCH": "Small",
    "SPREAD_FINGERS": "Big",
    "WAVE": "Hi there"
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
last_gesture = None
recognized_text = ""

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = classify_gesture(hand_landmarks.landmark)
            if gesture:
                recognized_text = gesture_dict[gesture]
                if recognized_text != last_gesture:
                    print(f"Recognized Gesture: {recognized_text}")
                    last_gesture = recognized_text

    # --- Caption panel (right side) ---
    caption_panel = np.ones((frame.shape[0], 320, 3), dtype=np.uint8) * 255
    cv2.putText(caption_panel, recognized_text, (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Combine webcam + caption panel
    combined = np.hstack((frame, caption_panel))
    cv2.imshow("Gesture Call Demo", combined)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
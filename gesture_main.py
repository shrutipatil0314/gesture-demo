import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture dictionary
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

def classify_gesture(landmarks):
    thumb_tip = landmarks[4].y
    index_tip = landmarks[8].y
    middle_tip = landmarks[12].y
    ring_tip = landmarks[16].y
    pinky_tip = landmarks[20].y
    wrist = landmarks[0].y

    # Helper tolerance
    def above(finger, offset=0.05): return finger < wrist - offset
    def below(finger, offset=0.05): return finger > wrist + offset

    # Specific gestures first
    if above(thumb_tip) and below(index_tip) and below(middle_tip):
        return "THUMBS_UP"
    if below(thumb_tip) and above(index_tip) and above(middle_tip):
        return "THUMBS_DOWN"
    if below(index_tip) and below(middle_tip) and below(ring_tip) and below(pinky_tip):
        return "FIST"
    if above(index_tip) and above(middle_tip) and below(ring_tip) and below(pinky_tip):
        return "PEACE"
    if abs(landmarks[4].x - landmarks[8].x) < 0.05 and abs(landmarks[4].y - landmarks[8].y) < 0.05:
        return "OK_SIGN"
    if above(index_tip) and below(middle_tip) and below(ring_tip) and below(pinky_tip):
        return "POINT_UP"
    if below(index_tip) and below(middle_tip) and below(ring_tip) and below(pinky_tip):
        return "POINT_DOWN"
    if above(index_tip) and abs(landmarks[8].x - landmarks[0].x) > 0.2:
        return "POINT_RIGHT"
    if above(index_tip) and abs(landmarks[0].x - landmarks[8].x) > 0.2:
        return "POINT_LEFT"
    if above(index_tip) and above(pinky_tip) and below(middle_tip) and below(ring_tip):
        return "ROCK_SIGN"
    if above(thumb_tip) and above(pinky_tip) and below(index_tip) and below(middle_tip):
        return "CALL_ME"
    if abs(landmarks[4].x - landmarks[8].x) < 0.02 and abs(landmarks[4].y - landmarks[8].y) < 0.02:
        return "PINCH"
    if above(index_tip) and above(middle_tip) and above(ring_tip) and above(pinky_tip) and abs(landmarks[8].x - landmarks[20].x) > 0.3:
        return "SPREAD_FINGERS"
    if above(index_tip) and above(middle_tip) and above(ring_tip) and above(pinky_tip):
        return "WAVE"
    if above(index_tip) and above(middle_tip) and above(ring_tip) and above(pinky_tip):
        return "OPEN_HAND"

    return None

# Webcam loop
cap = cv2.VideoCapture(0)
last_gesture = None

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
                text = gesture_dict[gesture]
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                if text != last_gesture:
                    print(f"Recognized Gesture: {text}")
                    last_gesture = text

    cv2.imshow("Gesture Recognition (15 Gestures)", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
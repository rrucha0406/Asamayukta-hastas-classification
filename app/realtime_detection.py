import cv2
import mediapipe as mp
import numpy as np
import joblib

# load model
model = joblib.load("models/mudra_model.pkl")
le = joblib.load("models/label_encoder.pkl")

# mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

# webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            # extract landmarks
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                prediction = model.predict([landmarks])
                gesture = le.inverse_transform(prediction)[0]

                # display text
                cv2.putText(frame, gesture, (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,255,0), 2)

            # draw hand
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Mudra Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
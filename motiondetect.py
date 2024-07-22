import cv2
import mediapipe as mp

# Initialize Mediapipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=8,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(frame_rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the resulting frame
        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
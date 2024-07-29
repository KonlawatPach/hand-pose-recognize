import cv2
import mediapipe as mp

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing_utils = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hand_detected = results.multi_hand_landmarks is not None

    if hand_detected:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box coordinates
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]

            # Convert normalized coordinates to pixel coordinates
            h, w, _ = image.shape
            x_coords = [int(x * w) for x in x_coords]
            y_coords = [int(y * h) for y in y_coords]

            # Calculate bounding box
            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            # Draw bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=None, connection_drawing_spec=drawing_utils)

    # Display the text on the image
    text = f'Hand Detected: {hand_detected}'
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the image
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

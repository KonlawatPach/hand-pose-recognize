import cv2
import mediapipe as mp
import csv

# Initialize Mediapipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

def normalize_landmarks(landmarks):
    # Using the wrist landmark (index 0) as the reference point
    wrist = landmarks[0]
    normalized = [(landmark.x - wrist.x, landmark.y - wrist.y, landmark.z - wrist.z) for landmark in landmarks]
    return normalized

with mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    
    # Open a CSV file to save landmark data
    with open('normalized_hand_landmarks.csv', 'w', newline='') as csvfile:
        # Assuming you will add a label column later
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['frame', 'hand', 'landmark', 'x', 'y', 'z'])  # Header
        frame_count = 0

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

            # Save hand landmarks to CSV
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    normalized_landmarks = normalize_landmarks(hand_landmarks.landmark)

                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        # Convert landmark points to pixel coordinates
                        height, width, _ = frame.shape
                        cx, cy = int(landmark.x * width), int(landmark.y * height)
                        
                        cv2.putText(frame, str(f"{idx} {normalized_landmarks[idx][0]:.2f} {normalized_landmarks[idx][1]:.2f}"), 
                            (cx, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            # for idx, landmark in enumerate(normalized_landmarks):
                            #     csvwriter.writerow([frame_count, hand_idx, idx, landmark[0], landmark[1], landmark[2]])

            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the resulting frame
            cv2.imshow('Hand Detection', frame)
            frame_count += 1

            if cv2.waitKey(5) & 0xFF == 27:
                break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()

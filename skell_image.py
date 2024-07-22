import cv2
import mediapipe as mp
import csv

def capture_and_detect_hands():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the live camera feed
        cv2.imshow('Camera', frame)

        # Capture image when spacebar is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                # Initialize CSV data
                landmark_list = []

                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw hand landmarks on the image
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract landmark coordinates
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        landmark_list.append([hand_idx, idx, landmark.x, landmark.y, landmark.z])

                # Save landmarks to CSV file
                with open('hand_landmarks.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Hand", "ID", "X", "Y", "Z"])
                    writer.writerows(landmark_list)

                print("Hand landmarks saved to hand_landmarks.csv")

            # Display the image with hand landmarks
            cv2.imshow('Hand Skeleton', frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_and_detect_hands()

# show class Label
import cv2
import numpy as np
import mediapipe as mp
import pickle
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# Initialize Mediapipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load Model & Scaler
with open('model-wash.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Start video capture
cap = cv2.VideoCapture(0)

def normalize_landmarks(landmarks):
    # Using the wrist landmark (index 0) as the reference point
    wrist = landmarks[0]
    normalized = [(landmark.x - wrist.x, landmark.y - wrist.y, landmark.z - wrist.z) for landmark in landmarks]
    return normalized

def display_class(frame, multi_hand_landmarks):
    input_x = []
    for ihand, hand_landmarks in enumerate(multi_hand_landmarks):
        handList = []
        for axis in range(0, 2):  #x, y, z
            normalized_landmarks = normalize_landmarks(hand_landmarks.landmark)
            for idx, landmark in enumerate(hand_landmarks.landmark):
                handList.append(normalized_landmarks[idx][axis])
        input_x.extend(handList)

        # duplicate อีกข้าง
        if(len(multi_hand_landmarks) <= 1):
            input_x.extend(handList)
    
    
    # loop check distance
    for i in range(1, 22):
        input_x.append(((input_x[i+42]-input_x[i])**2 + (input_x[i+63]-input_x[i+21])**2)**0.5)
    
    try:
        new_data = np.array(input_x).reshape(1, -1)
        new_data_normalized = scaler.transform(new_data)
        y_pred = model.predict(new_data_normalized)
        
        # Display Class
        cv2.putText(frame, str(f"Class : {y_pred[0]}"), 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
    except:
        print(input_x)

def open_camera():
    with mp_hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            height, width, _ = frame.shape
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
                for ihand, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    normalized_landmarks = normalize_landmarks(hand_landmarks.landmark)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                display_class(frame, results.multi_hand_landmarks)

            # Check Key Press
            key = cv2.waitKey(5) & 0xFF
            if key == 27: # Add +1 when Press Esc
                break

            # Display the resulting frame
            cv2.imshow('Hand Detection', frame)
                    

    # Release the capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    open_camera()

main()
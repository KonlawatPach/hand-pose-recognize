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


def add_csv_header(csvwriter):
    columntype = ["hand1-x", "hand1-y","hand2-x", "hand2-y","handdistance-"]
    header_row = ["y"]

    for ct in columntype:
        subrow = []
        for c in range(0, 21):
            subrow.append(ct + str(c))
            header_row.extend(subrow)
        header_row.extend(subrow)
    
    csvwriter.writerow(header_row)

def add_csv_body(csvwriter, classnum, multi_hand_landmarks):
    body_row = [classnum]

    # loop 2 hand, if only one will duplicate
    for ihand, hand_landmarks in enumerate(multi_hand_landmarks):
        handList = []
        for axis in range(0, 2):  #x, y, z
            normalized_landmarks = normalize_landmarks(hand_landmarks.landmark)
            for idx, landmark in enumerate(hand_landmarks.landmark):
                handList.append(normalized_landmarks[idx][axis])
        body_row.extend(handList)

        # duplicate อีกข้าง
        if(len(multi_hand_landmarks) <= 1):
            body_row.extend(handList)
    
    
    # loop check distance
    for i in range(1, 22):
        body_row.append(((body_row[i+42]-body_row[i])**2 + (body_row[i+63]-body_row[i+21])**2)**0.5)

    csvwriter.writerow(body_row)

def open_camera():
    countdown = -1
    classnum = 1

    with mp_hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
    
        # Open a CSV file to save landmark data
        with open('normalized_hand_landmarks.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            add_csv_header(csvwriter)

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

                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            # Convert landmark points to pixel coordinates
                            cx, cy = int(landmark.x * width), int(landmark.y * height)
                            cv2.putText(frame, str(f"{idx} {normalized_landmarks[idx][0]:.2f} {normalized_landmarks[idx][1]:.2f}"), 
                                (cx, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            if(idx == 0):
                                cv2.putText(frame, str(ihand+1), 
                                (cx, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display Class and CountDown
                cv2.putText(frame, str(classnum), 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                if(countdown>0):
                    cv2.putText(frame, str(countdown//10), 
                        (width-20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    countdown-=1

                if(countdown==0):
                    add_csv_body(csvwriter, classnum, results.multi_hand_landmarks)
                    countdown = -1

                # Check Key Press
                key = cv2.waitKey(5) & 0xFF
                if key == 27: # Add +1 when Press Esc
                    break
                elif key == ord('c'): # Add +1 Class when Press C
                    classnum = classnum+1 if classnum+1<=10 else 1
                elif key == 32: # Shutter when Press SpaceBar
                    countdown = 50

                # Display the resulting frame
                cv2.imshow('Hand Detection', frame)
                    

    # Release the capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    open_camera()

main()
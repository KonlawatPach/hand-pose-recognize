import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# Initialize Mediapipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Load Model & Scaler
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ล้างมือกี่วินาที
timeflame_success_perhand = 80
class_count = len(model.classes_)
hand_success = [0] * class_count

def add_hand_success(class_num):
    global hand_success
    hand_success[class_num-1] += 1
    print(hand_success)

def reset_hand_success():
    global hand_success
    hand_success = [0] * class_count

def check_hand_success(frame, width, height):
    global hand_success
    count_hand_success = 0

    for h in hand_success:
        if(h >= timeflame_success_perhand):
            count_hand_success += 1

    if(count_hand_success < class_count):
        frame = put_text_pil(frame, "ประสานอินไม่สะอาด\nไปประสานอินใหม่นะ", 
            (width//4.5, height//2), font_size=60, color=(255, 0, 0))
    else:
        frame = put_text_pil(frame, "ประสานอินได้เยี่ยมมาก", 
            (width//4.8, height//2), font_size=60, color=(0, 255, 0))
    return frame

def normalize_landmarks(landmarks):
    # Using the wrist landmark (index 0) as the reference point
    wrist = landmarks[0]
    normalized = [(landmark.x - wrist.x, landmark.y - wrist.y, landmark.z - wrist.z) for landmark in landmarks]
    return normalized

def put_text_pil(img, text, position, font_size=20, color=(0, 0, 255)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font_path = "WR Tish Kid.ttf"
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def check_class(frame, multi_hand_landmarks):
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
    for i in range(0, 21):
        input_x.append(((input_x[i+41]-input_x[i])**2 + (input_x[i+62]-input_x[i+20])**2)**0.5)
    
    try:
        new_data = np.array(input_x).reshape(1, -1)
        new_data_normalized = scaler.transform(new_data)
        y_pred = model.predict(new_data_normalized)

        add_hand_success(y_pred[0])
    except:
        print(input_x)

def open_camera():
    with mp_hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        countdown = -1
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
                if(countdown <= 0):
                    reset_hand_success()
                countdown = 80
                for ihand, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    normalized_landmarks = normalize_landmarks(hand_landmarks.landmark)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                check_class(frame, results.multi_hand_landmarks)
                
            if(countdown>0):
                cv2.putText(frame, str(countdown//10), 
                    (width-20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                countdown-=1

            if(countdown==0):
                frame = check_hand_success(frame, width, height)

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
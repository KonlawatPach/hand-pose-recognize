import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load data
data = pd.read_csv('normalized_hand_landmarks.csv')

# Separate features and labels
X = data.drop('y', axis=1).values
y = data['y'].values

# Normalize labels to start from 0
unique_labels = np.unique(y)
label_map = {label: index for index, label in enumerate(unique_labels)}
y = np.array([label_map[label] for label in y])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(len(unique_labels), activation='softmax')  # Number of unique classes in y
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

def predict_gesture(landmarks):
    # Normalize the landmarks
    landmarks = np.array(landmarks).reshape(1, -1)
    landmarks = scaler.transform(landmarks)
    
    # Predict the class
    prediction = model.predict(landmarks)
    return np.argmax(prediction, axis=1)[0]

# Example usage:
# landmarks = [list of normalized hand landmarks]
# gesture = predict_gesture(landmarks)
# print(f'Predicted gesture class: {gesture}')

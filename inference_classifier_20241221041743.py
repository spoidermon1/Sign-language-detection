import os
import pickle
import cv2
import mediapipe as mp

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the trained model and label encoder
with open('classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)

with open('label_encoder.pickle', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the camera is successfully opened
if not cap.isOpened():
    print("Error: Unable to access the camera.")
else:
    print("Camera successfully opened.")

# Function to process the hand landmarks and prepare the data for prediction
def extract_hand_landmarks(frame):
    data_aux = []
    x_ = []
    y_ = []

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Collect landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Only return data if landmarks are valid (length should be 42)
        if len(data_aux) == 42:
            return data_aux
    return None  # Return None if no landmarks were detected or data is invalid

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break
    
    # Extract hand landmarks and prepare data
    hand_data = extract_hand_landmarks(frame)

    if hand_data:
        # Predict the class using the trained classifier
        prediction = classifier.predict([hand_data])
        predicted_label = label_encoder.inverse_transform(prediction)

        # Display the predicted class
        cv2.putText(frame, f'Prediction: {predicted_label[0]}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with the predicted label
    cv2.imshow('frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

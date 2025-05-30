import os
import cv2
import string

# Directory to store the data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (A-Z, 26 letters)
alphabet_classes = list(string.ascii_uppercase)  # ['A', 'B', 'C', ..., 'Z']
dataset_size = 100  # Number of images per class

# Initialize webcam
cap = cv2.VideoCapture(0)

for letter in alphabet_classes:
    # Create a folder for each letter if it doesn't already exist
    letter_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)

    print(f'Collecting data for class {letter}')

    # Prompt user to prepare for data collection
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access webcam.")
            break
        cv2.putText(frame, f'Get ready for {letter}! Press "Q" to start.', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Start collecting data
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save the captured frame as an image
        file_path = os.path.join(letter_dir, f'{counter}.jpg')
        cv2.imwrite(file_path, frame)
        counter += 1

    print(f'Finished collecting data for class {letter}')

# Release resources
cap.release()
cv2.destroyAllWindows()

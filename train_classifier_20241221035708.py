import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the preprocessed data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Inspect and preprocess data
print("Inspecting data structure...")
try:
    # Detect and encode non-numerical elements
    processed_data = []
    for sample in data_dict['data']:
        processed_sample = []
        for element in sample:
            if isinstance(element, str):  # Handle non-numerical data
                # Encode strings as integers
                le = LabelEncoder()
                element = le.fit_transform([element])[0]
            processed_sample.append(float(element))
        processed_data.append(processed_sample)

    # Determine the maximum sequence length
    max_length = max(len(sample) for sample in processed_data)

    # Pad sequences to make them uniform in length
    data = pad_sequences(processed_data, maxlen=max_length, padding='post', dtype='float32')

    # Convert labels to a NumPy array
    labels = np.asarray(data_dict['labels'], dtype=np.int32)

    # Validate data and labels consistency
    assert len(data) == len(labels), "Mismatch between data and labels"

except Exception as e:
    print(f"Error in processing data: {e}")
    exit()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Initialize the Random Forest Classifier
model = RandomForestClassifier()

# Train the model
print("Training the model...")
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Evaluate the model's performance
score = accuracy_score(y_predict, y_test)
print(f"{score * 100:.2f}% of samples were classified correctly!")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_predict, zero_division=1))

# Save the trained model to a file
model_file = 'model.p'
with open(model_file, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"Model saved to {model_file}")

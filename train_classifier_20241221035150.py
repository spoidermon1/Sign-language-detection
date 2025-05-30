import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the preprocessed data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Inspect the data structure
print("Inspecting data structure...")
print(f"Type of data_dict['data']: {type(data_dict['data'])}")
print(f"Sample data: {data_dict['data'][:5]}")

# Check and preprocess data
try:
    # Convert each sample to a NumPy array and handle inconsistencies
    data = [np.asarray(sample, dtype=np.float32) for sample in data_dict['data']]
    data = np.array(data)

    # Ensure that all data samples have the same shape
    sample_shape = data[0].shape
    assert all(sample.shape == sample_shape for sample in data), "Inconsistent data shapes!"
except Exception as e:
    print(f"Error in processing data: {e}")
    exit()

# Convert labels to a NumPy array
labels = np.asarray(data_dict['labels'], dtype=np.int32)

# Validate data and labels consistency
if len(data) != len(labels):
    print("Mismatch between data and labels. Exiting.")
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

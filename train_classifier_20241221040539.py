import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
with open('data.pickle', 'rb') as f:
    dataset = pickle.load(f)

data = dataset['data']
labels = dataset['labels']

# Encode the labels (A-Z -> 0-25)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Convert to numpy arrays
data = np.array(data)
encoded_labels = np.array(encoded_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)

# Train a classifier (Random Forest in this case)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the classifier and label encoder to disk for later use
with open('classifier.pickle', 'wb') as f:
    pickle.dump(classifier, f)
    
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model and label encoder saved!")

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
    data = pad_sequences(processed_data, maxlen=max_length, padding='post', dtype='float3

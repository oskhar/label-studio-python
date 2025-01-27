
import os
import numpy as np
import librosa
import tensorflow as tf
from label_studio_sdk import Client

# Step 1: Connect to Label Studio
label_studio_url = "http://localhost:8080"
api_key = "23ce5b08cc51665ab3bb19e0bf0c43b21c7c0723"
client = Client(url=label_studio_url, api_key=api_key)

# Step 2: Get project from Label Studio
project_id = 8
project = client.get_project(project_id)

# Step 3: Fetch annotations and prepare data
tasks = project.get_tasks()
x = []  # features (mfccs)
y = []  # labels

audio_path_root = "/home/oskhar/.local/share/label-studio/media/"

for task in tasks:
    audio_path_tmp = task['data']['audio']  # Path to audio file
    audio_path = audio_path_root + audio_path_tmp.replace("/data/", "")

    labels = task['data']['label']  # Annotations/labels
    label_data = eval(labels)  # Convert JSON-like string to Python dictionary

    # Extract MFCC features
    try:
        audio, sr = librosa.load(audio_path, sr=None)  # Load audio with its original sampling rate
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # Extract MFCC
        mfcc_mean = np.mean(mfcc.T, axis=0)  # Take mean across time axis

        # Append to datasets
        x.append(mfcc_mean)
        y.append(label_data[0]["labels"][0])  # Example: Extract the first label
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")

# Step 4: Prepare data for TensorFlow
x = np.array(x)
y = np.array(y)

# Convert labels to one-hot encoding
unique_labels = list(set(y))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y_encoded = np.array([label_to_index[label] for label in y])

# Step 5: Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(40,)),  # Input shape matches MFCC feature size
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(unique_labels), activation='softmax')  # Output matches number of unique labels
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
model.fit(x, y_encoded, epochs=20, batch_size=32)

# Step 7: Save the model
model.save("audio_recognition_model.h5")

print("Model training complete and saved as 'audio_recognition_model.h5'.")

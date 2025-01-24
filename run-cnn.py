import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pandas as pd

# Load Label Studio annotations (from the CSV file you provided)
csv_file = 'label-studio-file.csv'
annotations = pd.read_csv(csv_file)

# Function to extract mel-spectrograms
def extract_mel_spectrogram(audio_path, sr=22050, n_mels=128):
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect_db

# Prepare dataset
audio_files = []
labels = []
for _, row in annotations.iterrows():
    audio_path = row['audio']  # Path to the audio file
    label = row['label']  # Label from Label Studio

    # Convert label from JSON string to a numeric label (adjust this as needed)
    # Assuming label is something like [{"labels": ["ClassName"]}]
    label_class = eval(row['label'])[0]['labels'][0]

    print(eval(row))

    # Load audio and extract mel-spectrogram
    mel_spectrogram = extract_mel_spectrogram(audio_path)

    # Resize spectrogram to a fixed shape (e.g., 128x128)
    mel_spectrogram = librosa.util.fix_length(mel_spectrogram, size=128)

    # Append to the dataset
    audio_files.append(mel_spectrogram)
    labels.append(label_class)

# Convert to numpy arrays and one-hot encode labels
X_data = np.array(audio_files)
y_labels = to_categorical(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

# Reshape data to fit CNN input (e.g., 128x128x1 for grayscale)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(np.unique(labels)), activation='softmax')  # Number of unique classes
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

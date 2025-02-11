import os
import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd
from datetime import datetime
from label_studio_sdk import Client

# ===========================
# 🔹 1. Koneksi ke Label Studio
# ===========================
label_studio_url = "http://localhost:8080"
api_key = "6b1e655ca53d8d10366d3d7fcb6940e4d072c4de"
client = Client(url=label_studio_url, api_key=api_key)

# Fetch project
project_id = 2
project = client.get_project(project_id)
annotations = project.get_tasks()
audio_path_root = "/home/oskhar/.local/share/label-studio/media/"

# ===========================
# 🔹 2. Fungsi Ekstraksi MFCC
# ===========================
def extract_mfcc(audio_path, start, end, sample_rate=16000, n_mfcc=40, max_len=128):
    """
    Ekstraksi MFCC dari segmen audio dengan padding atau cropping.
    Jika file tidak ditemukan atau error, langsung dilewati.
    """
    try:
        if not os.path.exists(audio_path):
            print(f"❌ File tidak ditemukan: {audio_path}")
            return None

        # Load audio
        audio = AudioSegment.from_file(audio_path)
        segment = audio[start * 1000:end * 1000]

        # Convert to numpy array
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)

        # Resample audio
        samples_resampled = librosa.resample(samples, orig_sr=segment.frame_rate, target_sr=sample_rate)

        # Extract MFCC
        mfcc_features = librosa.feature.mfcc(y=samples_resampled, sr=sample_rate, n_mfcc=n_mfcc)

        # Padding or cropping MFCC
        if mfcc_features.shape[1] < max_len:
            pad_width = max_len - mfcc_features.shape[1]
            mfcc_features = np.pad(mfcc_features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc_features = mfcc_features[:, :max_len]

        return mfcc_features
    except Exception as e:
        print(f"⚠️ Error processing {audio_path} from {start} to {end}: {e}")
        return None

# ===========================
# 🔹 3. Proses Data: Filter File yang Tidak Ditemukan
# ===========================
audio_files = []
labels = []
label_mapping = {}
label_counter = 0
i = 1

for row in annotations:
    audio_path_tmp = row['data']['audio']
    audio_path = os.path.join(audio_path_root, audio_path_tmp.replace("/data/", ""))

    # ✅ Cek apakah file ada, jika tidak langsung skip
    if not os.path.exists(audio_path):
        print(f"🚨 Skipping missing file: {audio_path}")
        continue

    label_data_tmp = row["annotations"][0]["result"]

    print(f"Processing: {i}/{len(annotations)} - {audio_path}")
    i += 1

    mad_segments = []
    for annotation in label_data_tmp:
        start = annotation["value"]["start"]
        end = annotation["value"]["end"]
        label_name = annotation["value"]["labels"][0]

        # Assign unique integer to each label
        if label_name not in label_mapping:
            label_mapping[label_name] = label_counter
            label_counter += 1

        mfcc = extract_mfcc(audio_path, start, end)
        if mfcc is not None and mfcc.shape == (40, 128):
            audio_files.append(mfcc)
            labels.append(label_mapping[label_name])
            mad_segments.append((start, end))

    # ✅ Skip jika tidak ada data valid dari file ini
    if len(mad_segments) == 0:
        print(f"⚠️ No valid MAD data found in {audio_path}, skipping...")
        continue

    try:
        # Load full audio
        full_audio = AudioSegment.from_file(audio_path)
        duration = len(full_audio) / 1000  # Convert to seconds
    except Exception as e:
        print(f"🚨 Error loading full audio {audio_path}: {e}")
        continue  # Skip jika file korup

    # Generate non-MAD segments
    non_mad_segments = []
    prev_end = 0
    for start, end in mad_segments:
        if start - prev_end > 2:
            non_mad_segments.append((prev_end, start))
        prev_end = end

    if duration - prev_end > 2:
        non_mad_segments.append((prev_end, duration))

    for start, end in non_mad_segments:
        mfcc = extract_mfcc(audio_path, start, end)
        if mfcc is not None and mfcc.shape == (40, 128):
            audio_files.append(mfcc)
            labels.append(label_mapping.setdefault("Non-MAD", label_counter))

# ✅ Pastikan hanya memasukkan data valid
if len(audio_files) == 0:
    raise ValueError("🚨 Tidak ada data valid setelah filtering. Pastikan file tersedia dan tidak korup.")

X_data = np.stack(audio_files)
y_labels = to_categorical(labels, num_classes=len(label_mapping))

# ===========================
# 🔹 4. Split Dataset
# ===========================
X_train, X_tmp, y_train, y_tmp = train_test_split(X_data, y_labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# ===========================
# 🔹 5. Definisi Model CNN
# ===========================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(40, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Regularisasi
    Dense(len(label_mapping), activation='softmax')  # Output sesuai jumlah kelas
])

# Kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===========================
# 🔹 6. Pelatihan Model
# ===========================
start_time = datetime.now()
print(f"Training started at: {start_time}")

history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_val, y_val),
    batch_size=32
)

end_time = datetime.now()
print(f"Training ended at: {end_time}")
print(f"Training duration: {end_time - start_time}")

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Simpan model
model.save("cnn_audio_recognition.h5")

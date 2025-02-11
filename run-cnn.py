import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from label_studio_sdk import Client

# 1. Connect to Label Studio
label_studio_url = "http://localhost:8080"
api_key = "23ce5b08cc51665ab3bb19e0bf0c43b21c7c0723"
client = Client(url=label_studio_url, api_key=api_key)

# 2. Dapatkan proyek Label Studio berdasarkan ID proyek
project_id = 8  # Ganti dengan ID proyek Anda
project = client.get_project(project_id)

# 3. Ambil anotasi dari proyek
tasks = project.get_tasks()

```
Memuat file audio dari audio_path menggunakan librosa.load, dengan sampling rate (sr) yang dapat ditentukan pengguna. Selanjutnya, kode menghasilkan spektrogram Mel (mel_spect) menggunakan librosa.feature.melspectrogram, yang mengubah sinyal audio menjadi representasi frekuensi berbasis Mel dengan jumlah filter bank tertentu (n_mels). Setelah itu, spektrogram Mel yang masih dalam bentuk daya diubah ke skala desibel (mel_spect_db) menggunakan librosa.power_to_db, dengan referensi normalisasi berdasarkan nilai maksimum. Hasil akhirnya adalah spektrogram Mel dalam skala desibel, yang dapat digunakan untuk analisis lebih lanjut, seperti pemrosesan suara atau pembelajaran mesin.
```
# Function to extract mel-spectrograms
def extract_mel_spectrogram(audio_path, sr=22050, n_mels=128):
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect_db

```
Analisis sinyal audio dengan menghasilkan spektrogram Mel dalam skala desibel. Proses dimulai dengan memuat file audio dari audio_path menggunakan librosa.load, di mana pengguna dapat menentukan sampling rate (sr). Kemudian, sinyal audio dikonversi menjadi spektrogram Mel (mel_spect) menggunakan librosa.feature.melspectrogram, yang merepresentasikan energi frekuensi dalam skala Mel dengan jumlah filter tertentu (n_mels). Setelah itu, librosa.power_to_db digunakan untuk mengubah nilai daya dalam spektrogram menjadi skala desibel (mel_spect_db), dengan normalisasi berdasarkan nilai maksimum. Hasil akhirnya adalah representasi visual dari konten frekuensi dalam sinyal audio, yang berguna untuk berbagai aplikasi seperti analisis suara, pengenalan pola, atau pembelajaran mesin.
```
# Prepare dataset
audio_files = []
labels = []
for task in tasks:
    annotations = task.get("annotations", [])
    if not annotations:
        continue

    # Ambil path audio dan label dari anotasi
    audio_path = task['data']['audio']  # Sesuaikan dengan nama field di Label Studio
    label = annotations[0]['result'][0]['value']['choices'][0]  # Sesuaikan dengan struktur anotasi Anda

    # Load audio dan extract mel-spectrogram
    mel_spectrogram = extract_mel_spectrogram(audio_path)

    # Resize spectrogram to a fixed shape (e.g., 128x128)
    mel_spectrogram = librosa.util.fix_length(mel_spectrogram, size=128)

    # Append to the dataset
    audio_files.append(mel_spectrogram)
    labels.append(label)

```
Mengonversi label kategori menjadi nilai numerik agar dapat digunakan dalam pemrosesan data, seperti dalam model machine learning. Prosesnya dimulai dengan mengambil daftar unik dari label yang ada menggunakan `set(labels)`, lalu mengubahnya kembali menjadi daftar (`unique_labels`). Selanjutnya, kode membuat pemetaan dari setiap label unik ke indeks numerik dengan `enumerate()`, menghasilkan dictionary `label_to_index`. Terakhir, daftar `labels` asli dikonversi ke daftar angka (`numeric_labels`) dengan menggantikan setiap label dengan indeks yang sesuai dari dictionary. Hasil akhirnya adalah representasi numerik dari label, yang memudahkan analisis atau pelatihan model.
```
# Convert labels to numeric values
unique_labels = list(set(labels))
label_to_index = {label: i for i, label in enumerate(unique_labels)}
numeric_labels = [label_to_index[label] for label in labels]

```
Konversi data audio dan label kategori ke dalam format yang sesuai untuk pemrosesan dalam model machine learning. Pertama, `audio_files` yang berisi data audio dikonversi menjadi array NumPy (`X_data = np.array(audio_files)`) agar dapat digunakan dalam komputasi yang lebih efisien. Selanjutnya, label numerik yang telah dikonversi sebelumnya (`numeric_labels`) diubah menjadi representasi one-hot encoding menggunakan `to_categorical(numeric_labels)`. One-hot encoding mengubah setiap label menjadi vektor biner, di mana hanya satu elemen yang bernilai 1 dan sisanya 0, memungkinkan model untuk memahami hubungan kategori dengan lebih baik. Hasil akhirnya adalah `X_data` dalam bentuk array NumPy dan `y_labels` dalam format one-hot encoding, siap untuk digunakan dalam pelatihan model machine learning.
```
# Convert to numpy arrays and one-hot encode labels
X_data = np.array(audio_files)
y_labels = to_categorical(numeric_labels)

```
membagi dataset menjadi data pelatihan dan pengujian menggunakan fungsi train_test_split dari sklearn.model_selection. X_data berisi data fitur (audio yang telah diproses), sedangkan y_labels berisi label dalam format one-hot encoding. Parameter test_size=0.2 menentukan bahwa 20% dari data akan digunakan sebagai data pengujian, sementara 80% sisanya digunakan untuk pelatihan. Parameter random_state=42 memastikan bahwa pembagian data bersifat deterministik, sehingga hasil yang diperoleh akan selalu sama setiap kali kode dijalankan. Hasil akhirnya adalah empat variabel: X_train dan y_train untuk pelatihan model, serta X_test dan y_test untuk evaluasi model.
```
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
    Dense(len(unique_labels), activation='softmax')  # Number of unique classes
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

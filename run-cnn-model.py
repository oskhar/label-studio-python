
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from datetime import datetime
import matplotlib.pyplot as plt
from pydub import AudioSegment
import librosa

# Load model yang sudah disimpan
model = load_model("cnn_audio_recognition.keras")

def extract_mfcc(audio_path, start, end, sample_rate=16000, n_mfcc=40, max_len=128):
    """
    Ekstraksi MFCC dari segmen audio dengan padding atau cropping.
    """
    try:
        # Load audio menggunakan pydub
        audio = AudioSegment.from_file(audio_path)
        # Potong segmen berdasarkan waktu (dalam milidetik)
        segment = audio[start * 1000:end * 1000]
        # Konversi audio ke numpy array
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        # Resample audio jika diperlukan
        samples_resampled = librosa.resample(samples, orig_sr=segment.frame_rate, target_sr=sample_rate)
        # Ekstraksi MFCC
        mfcc_features = librosa.feature.mfcc(y=samples_resampled, sr=sample_rate, n_mfcc=n_mfcc)
        # Pad atau crop MFCC agar memiliki dimensi tetap
        if mfcc_features.shape[1] < max_len:
            pad_width = max_len - mfcc_features.shape[1]
            mfcc_features = np.pad(mfcc_features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc_features = mfcc_features[:, :max_len]
        return mfcc_features
    except Exception as e:
        print(f"Error processing {audio_path} from {start} to {end}: {e}")
        return None

def detect_mad(model, audio_path, segment_duration=1, sample_rate=16000):
    """
    Deteksi jenis mad dan posisi dalam audio.

    Parameters:
    - model: Model yang telah dilatih.
    - audio_path: Path ke file audio.
    - segment_duration: Durasi setiap segmen dalam detik.
    - sample_rate: Sample rate untuk ekstraksi audio.

    Returns:
    - results: List deteksi mad dengan jenis dan posisi (start, end).
    """
    try:
        # Load audio menggunakan pydub
        audio = AudioSegment.from_file(audio_path)
        audio_duration = len(audio) / 1000  # Durasi audio dalam detik

        results = []

        # Proses audio dalam segmen-segmen
        for start in range(0, int(audio_duration), segment_duration):
            end = start + segment_duration
            mfcc_features = extract_mfcc(audio_path, start, end, sample_rate=sample_rate)

            if mfcc_features is not None:
                # Reshape agar sesuai dengan input model
                mfcc_features = mfcc_features[..., np.newaxis]
                mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Tambahkan batch dimension

                # Prediksi dengan model
                predictions = model.predict(mfcc_features)
                predicted_label = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions, axis=1)[0]

                # Tambahkan hasil ke list
                results.append({
                    "start": start,
                    "end": end,
                    "mad_type": predicted_label,
                    "confidence": confidence
                })

        return results

    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return []

# Contoh penggunaan
audio_path = "/home/oskhar/.local/share/label-studio/media/upload/8/e103702e-muammar-z183.mp3"
segment_duration = 1  # Durasi segmen dalam detik

detection_results = detect_mad(model, audio_path, segment_duration=segment_duration)

# Print hasil deteksi
if detection_results:
    print("Deteksi Mad:")
    for result in detection_results:
        print(f"Start: {result['start']}s, End: {result['end']}s, Mad Type: {result['mad_type']}, Confidence: {result['confidence']:.2f}")
else:
    print("No mad detected or processing failed.")

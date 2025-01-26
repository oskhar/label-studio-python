# Gunakan image TensorFlow sebagai base image
FROM tensorflow/tensorflow:latest

# Set direktori kerja di dalam container
WORKDIR /app

# Salin seluruh file dari direktori lokal ke dalam container
COPY . /app

# Install semua package yang diperlukan
RUN pip install --upgrade pip && \
    pip install numpy librosa matplotlib scikit-learn label-studio-sdk

# Default command untuk menjalankan container
CMD ["python3"]

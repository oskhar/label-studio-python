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
api_key = "6b1e655ca53d8d10366d3d7fcb6940e4d072c4de"
client = Client(url=label_studio_url, api_key=api_key)

# 2. Dapatkan proyek Label Studio berdasarkan ID proyek
project_id = 2  # Ganti dengan ID proyek Anda
project = client.get_project(project_id)

# 3. Ambil anotasi dari proyek
tasks = project.get_tasks()

for task in tasks:
    print(task)

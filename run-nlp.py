
import label_studio_sdk
from label_studio_sdk import Client
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 1. Connect to Label Studio
label_studio_url = "http://localhost:8080"
api_key = "23ce5b08cc51665ab3bb19e0bf0c43b21c7c0723"
client = Client(url=label_studio_url, api_key=api_key)

# 2. Dapatkan proyek Label Studio berdasarkan ID proyek
project_id = 6  # Ganti dengan ID proyek Anda
project = client.get_project(project_id)

# 3. Ambil data anotasi dari proyek
tasks = project.get_tasks()

# 4. Persiapkan data untuk pelatihan
texts = []
sentiments = []

# Iterasi melalui tugas dan ambil teks dan label, pastikan anotasi ada
for task in tasks:
    text = task['data'].get('text')  # Teks yang dianotasi
    annotations = task.get('annotations', [])

    # Periksa apakah anotasi ada dan teks tidak kosong
    if annotations and text.strip():
        sentiment = annotations[0]['result'][0]['value']['choices'][0]  # Label sentiment
        texts.append(text)
        sentiments.append(sentiment)
    else:
        # Jika anotasi kosong atau teks kosong, melewati tugas
        print(f"Task {task['id']} belum diberi label atau teks kosong, melewati...")
        continue

# 5. Pastikan ada data sebelum melanjutkan
if len(texts) == 0:
    raise ValueError("Tidak ada data valid yang ditemukan setelah pengecekan. Pastikan ada teks yang dianotasi.")

# Buat DataFrame untuk memudahkan analisis dan pelatihan
data = pd.DataFrame({
    'text': texts,
    'sentiment': sentiments
})

# 6. Preprocessing teks
X = data['text']
y = data['sentiment']

# Konversi teks menjadi fitur menggunakan CountVectorizer
vectorizer = CountVectorizer(stop_words='english')  # Hilangkan stop words
X_vect = vectorizer.fit_transform(X)

# 7. Pembagian data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# 8. Pelatihan model dengan Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# 9. Evaluasi model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 10. Menyimpan model
import joblib
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

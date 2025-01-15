import pickle
import os

def load_model_and_vectorizer(model_path, vectorizer_path):
    """
    Memuat model dan vectorizer dari file pickle.
    """
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("[INFO] Vectorizer loaded successfully.")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("[INFO] Model loaded successfully.")

        return model, vectorizer
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        exit(1)
    except pickle.UnpicklingError as e:
        print(f"[ERROR] Failed to load pickle file: {e}")
        exit(1)

def predict_sentiment(model, vectorizer, texts):
    """
    Memprediksi sentimen untuk daftar teks.
    """
    try:
        # Transformasi teks menjadi vektor
        vectors = vectorizer.transform(texts)

        # Prediksi sentimen
        predictions = model.predict(vectors)
        return predictions
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        exit(1)

def main():
    # Path ke file pickle
    model_path = "sentiment_model.pkl"
    vectorizer_path = "vectorizer.pkl"

    # Memuat model dan vectorizer
    sentiment_model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

    # Contoh teks baru
    new_texts = [
        "I love this product! It's amazing.",
        "The experience was terrible, I hated it.",
        "The weather today is okay, nothing special."
    ]

    # Prediksi sentimen
    predictions = predict_sentiment(sentiment_model, vectorizer, new_texts)

    # Tampilkan hasil prediksi
    print("\n[RESULTS]")
    for text, sentiment in zip(new_texts, predictions):
        print(f"Text: {text} -> Sentiment: {sentiment}")

if __name__ == "__main__":
    main()

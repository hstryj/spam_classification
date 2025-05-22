import joblib
import re

def load_model_and_vectorizer(model_path='../models/spam_classifier_model.joblib',
                              vectorizer_path='../models/vectorizer.joblib'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_message(message, model, vectorizer):
    """
    Predykcja czy wiadomość jest spamem czy nie.
    """
    # Preprocessing wiadomości
    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)

    # Transformacja na wektor
    message_vec = vectorizer.transform([message])

    # Predykcja
    prediction = model.predict(message_vec)

    return "Spam" if prediction[0] == 1 else "Ham"
# 📬 Klasyfikacja wiadomości SMS jako spam lub ham (Naive Bayes)

Ten projekt implementuje prosty system wykrywania spamu w wiadomościach SMS z wykorzystaniem algorytmów sztucznej inteligencji. Wykorzystano probabilistyczny model Naive Bayes do klasyfikacji binarnej (spam/ham), z zastosowaniem czyszczenia tekstu, ekstrakcji cech metodą bag-of-words, strojenia hiperparametrów i oceny skuteczności modelu.

---

## 🧠 Technologie

- Python 3.9+
- Anaconda (zalecane środowisko)
- scikit-learn
- nltk
- pandas, numpy
- joblib
- matplotlib (opcjonalnie – do wykresów)

---

## 🗂️ Struktura projektu
spam_classification/
│
├── data/                     # Zbiór danych (np. SMSSpamCollection)
├── models/                   # Zapisany model i wektoryzator
├── notebooks/                # Notebook z trenowaniem i ewaluacją
├── src/                      # Moduły źródłowe
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── predict.py
├── README.md
└── requirements.txt

---

## ⚙️ Konfiguracja projektu

1. **Zainstaluj Anacondę**:  
   [https://www.anaconda.com/download](https://www.anaconda.com/download)

2. **Utwórz środowisko** (np. `spam_env`):

```bash
conda create -n spam_env python=3.9
conda activate spam_env

3. Zainstaluj wymagane biblioteki:
pip install -r requirements.txt

4. Pobierz zasoby NLTK (tylko za pierwszym razem) :
import nltk
nltk.download('stopwords')
nltk.download('punkt')

## 🚀 Uruchomienie projektu
	1.	Otwórz notebook:
    notebooks/spam_classification.ipynb
	2.	Uruchom komórki po kolei:
	•	Wczytywanie i czyszczenie danych
	•	Trenowanie i ocena modelu
	•	Zapis modelu i wektoryzatora do /models/
    3.	Przykład użycia predykcji:
    from src.predict import load_model_and_vectorizer, predict_message

model, vectorizer = load_model_and_vectorizer()
print(predict_message("You won a free ticket!", model, vectorizer))  # → Spam

## 🧪 Przykładowy wynik
Accuracy: 0.981
Macierz pomyłek:
[[964   2]
 [ 19 130]]

## 📦 Pliki wyjściowe
Po treningu modelu w katalogu models/ zapisują się:
	•	spam_classifier_model.joblib – wytrenowany model Naive Bayes
	•	vectorizer.joblib – wektoryzator (CountVectorizer)

🤖 Zastosowany algorytm
	•	Naive Bayes Classifier (MultinomialNB) – nadzorowany, probabilistyczny algorytm klasyfikacji
	•	Zadanie: klasyfikacja binarna (spam = 1, ham = 0)
	•	Czyszczenie tekstu: lowercase, tokenizacja, usuwanie stopwordów, stemming
	•	Ekstrakcja cech: bag-of-words z n-gramami (1,2)
	•	Strojenie hiperparametrów: GridSearchCV (alpha)

📚 Zbiór danych
Zbiór danych: SMS Spam Collection Dataset https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset




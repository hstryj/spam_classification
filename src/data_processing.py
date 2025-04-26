import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

def load_data(filepath):
    """
    Wczytuje dane SMS Spam Collection z pliku tekstowego.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Plik nie został znaleziony: {filepath}")

    df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
    return df

def basic_info(df):
    print("Pierwsze 5 wiadomości:")
    print(df.head())
    print("\nStatystyki labeli:")
    print(df['label'].value_counts())
    print("\nInformacje o danych:")
    print(df.info())

def vectorize_messages(messages):
    """
    Wektoruje wiadomości tekstowe do postaci numerycznej przy użyciu CountVectorizer.
    """
    vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
    X = vectorizer.fit_transform(messages)
    return X, vectorizer

def preprocess_messages(df):
    df["message"] = df["message"].str.lower()
    df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))
    df["message"] = df["message"].apply(lambda x: x.split())
    stop_words = set(stopwords.words("english"))
    df["message"] = df["message"].apply(lambda x: [word for word in x if word not in stop_words])
    stemmer = PorterStemmer()
    df["message"] = df["message"].apply(lambda x: [stemmer.stem(word) for word in x])
    df["message"] = df["message"].apply(lambda x: " ".join(x))
    return df
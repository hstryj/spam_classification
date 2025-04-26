import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

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
from sklearn.feature_extraction.text import CountVectorizer

def vectorize_messages(messages):
    vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
    X = vectorizer.fit_transform(messages)
    return X, vectorizer
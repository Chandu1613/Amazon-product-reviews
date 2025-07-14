from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    features = vectorizer.fit_transform(texts)
    return features, vectorizer
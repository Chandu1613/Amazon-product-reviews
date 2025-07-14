import streamlit as st
import joblib
from src.text_preprocessor import clean_text

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

model = joblib.load("model/best_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.title("📊 Product Review Sentiment Analyzer")

review = st.text_area("Enter a product review:", height=150)

if st.button("Analyze"):
    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😠"
    st.success(f"Sentiment: {sentiment}")
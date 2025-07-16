import streamlit as st
import joblib
from src.text_preprocessor import clean_text

# App Configuration
st.set_page_config(page_title="🧠 Sentiment Analyzer", layout="centered")

# Load model and vectorizer
model = joblib.load("model/best_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.markdown(
    """
    <h1 style='text-align: center;'>📝 Product Review Sentiment Analyzer</h1>
    <p style='text-align: center; font-size: 18px;'>Enter a product review below and find out if it's 💚 Positive or 💔 Negative using machine learning.</p>
    <hr>
    """,
    unsafe_allow_html=True
)

review = st.text_area("🗣️ Write your product review here:", height=150, placeholder="e.g., This camera is amazing! The picture quality is stunning...")

if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review before analyzing.")
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("✅ **Sentiment: Positive** 😊\nThis review expresses a positive opinion.")
            st.balloons()
        else:
            st.error("❌ **Sentiment: Negative** 😠\nThis review expresses a negative opinion.")
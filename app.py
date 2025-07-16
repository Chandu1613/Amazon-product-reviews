from flask import Flask, render_template, request
import joblib
from src.text_preprocessor import clean_text  # your cleaning logic

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/best_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        review = request.form["review"]
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😠"
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
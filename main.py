from src.data_loader import load_and_clean_data
from src.text_preprocessor import preprocess
from src.feature_extractor import vectorize_text
from src.model_trainer import train_models, evaluate_models

from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load data
df = load_and_clean_data("Amazon-reviews-limpo.csv")

# Step 2: Preprocess text
df = preprocess(df)

# Step 3: Feature extraction
X, vectorizer = vectorize_text(df['cleaned_review'])
y = df['label']

# Step 4: Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train models
models = train_models(X_train, y_train)

# Step 6: Evaluate and select best
best_model, best_name = evaluate_models(models, X_test, y_test)
print(f"\n✅ Best Model: {best_name}")

# Step 7: Save best model and vectorizer        
joblib.dump(best_model, "model/best_model.pkl",compress=9)
joblib.dump(vectorizer, "model/vectorizer.pkl",compress=6)
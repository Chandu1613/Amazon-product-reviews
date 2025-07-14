from src.data_loader import load_data
from src.text_preprocessor import preprocess
from src.feature_extractor import extract_features
from src.model_trainer import train_models
from src.evaluator import evaluate_models
from sklearn.model_selection import train_test_split

# 1. Load and filter data
df = load_data("data/reviews.csv")
df = df[df['Score'] != 3]
df['label'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)

# 2. Preprocess
df = preprocess(df)

# 3. Features
X, vectorizer = extract_features(df['text_clean'])
y = df['label']

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train multiple models
models = train_models(X_train, y_train)

# 6. Evaluate
f1_scores = evaluate_models(models, X_test, y_test)

# 7. Final result
best_model = max(f1_scores, key=f1_scores.get)
print(f"\n✅ Best model based on F1 score: {best_model} with score {f1_scores[best_model]:.4f}")

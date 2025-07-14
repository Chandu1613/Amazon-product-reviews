from sklearn.metrics import classification_report

def evaluate_models(models, X_test, y_test):
    scores = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        f1 = report["weighted avg"]["f1-score"]
        scores[name] = f1
        print(f"\n{name} Classification Report:\n")
        print(classification_report(y_test, preds))
    return scores
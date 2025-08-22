import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# Мини-датасет (можно расширить, чтобы лучше работало)
data = [
    ("The book on the table is mine.", 1),
    ("Mary seems to have been sleeping.", 1),
    ("The cat chased the mouse.", 1),
    ("It appears that John left.", 1),
    ("This sentence is perfectly fine.", 1),
    ("*Is the John tall?", 0),
    ("*She appears that is happy.", 0),
    ("*They are interested with linguistics.", 0),
    ("*Him likes they.", 0),
    ("*The child seems that is hungry.", 0),
    # Можно добавить ещё предложений, чтобы модель была точнее
]

df = pd.DataFrame(data, columns=["sentence","label"])
X_train, X_test, y_train, y_test = train_test_split(df["sentence"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=2000, n_jobs=1))
])

model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1  = f1_score(y_test, preds)
print(f"Accuracy: {acc:.3f}  F1: {f1:.3f}")

joblib.dump(model, "model.joblib")
print("✅ Model saved as model.joblib")

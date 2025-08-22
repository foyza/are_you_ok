# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# Загружаем CoLA из локальных файлов
train = pd.read_csv("data/train.tsv", sep="\t", header=None, names=["source","label","notes","sentence"])
dev   = pd.read_csv("data/dev.tsv", sep="\t", header=None, names=["source","label","notes","sentence"])

df = pd.concat([train[["sentence","label"]], dev[["sentence","label"]]], ignore_index=True)
df["sentence"] = df["sentence"].astype(str)
df["label"] = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(df["sentence"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,3), sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=3000, n_jobs=1))
])

model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1  = f1_score(y_test, preds)
print(f"✅ Accuracy: {acc:.3f}  F1: {f1:.3f}")

joblib.dump(model, "model.joblib")
print("✅ Model saved as model.joblib")

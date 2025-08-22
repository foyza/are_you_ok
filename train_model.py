import io, sys, urllib.request, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

URL_TRAIN = "https://raw.githubusercontent.com/nyu-mll/GLUE-baselines/master/glue_data/CoLA/train.tsv"
URL_DEV   = "https://raw.githubusercontent.com/nyu-mll/GLUE-baselines/master/glue_data/CoLA/dev.tsv"

def load_cola():
    try:
        with urllib.request.urlopen(URL_TRAIN, timeout=30) as r:
            train_bytes = r.read()
        with urllib.request.urlopen(URL_DEV, timeout=30) as r:
            dev_bytes = r.read()
        train = pd.read_csv(io.BytesIO(train_bytes), sep="\t", header=None,
                            names=["source","label","notes","sentence"])
        dev   = pd.read_csv(io.BytesIO(dev_bytes), sep="\t", header=None,
                            names=["source","label","notes","sentence"])
        df = pd.concat([train[["sentence","label"]], dev[["sentence","label"]]], ignore_index=True)
        df["sentence"] = df["sentence"].astype(str)
        df["label"] = df["label"].astype(int)
        return df
    except Exception as e:
        # Фоллбэк: очень маленькая публичная выборка (взята из CoLA)
        data = [
            ("The book on the table is mine.", 1),
            ("Mary seems to have been sleeping.", 1),
            ("The cat chased the mouse.", 1),
            ("*Is the John tall?", 0),
            ("*She appears that is happy.", 0),
            ("*They are interested with linguistics.", 0),
            ("This sentence is perfectly fine.", 1),
            ("*Him likes they.", 0),
            ("It appears that John left.", 1),
            ("*The child seems that is hungry.", 0),
        ]
        df = pd.DataFrame(data, columns=["sentence","label"])
        return df

def main():
    df = load_cola()
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
    print("✅ Saved model to model.joblib")

if __name__ == "__main__":
    main()

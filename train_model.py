import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

good_sentences = [
    "Сегодня хорошая погода.",
    "Я люблю программировать на Python.",
    "Машинное обучение — это интересно.",
    "Мы идём в кино вечером.",
    "У меня есть домашнее задание.",
    "Она купила новую книгу.",
    "Я читаю статью про AI.",
    "Моя собака любит играть на улице.",
    "Он учится в университете.",
    "Мы будем путешествовать летом.",
] * 20  

bad_sentences = [
    "sjandvwbsid qweqwe.",
    "lkmnvois dksjfhg.",
    "qwoeirupz mxncvb.",
    "asdkljfh qweoiu.",
    "zxcmnqwe lkjhgf.",
] * 40  

data = good_sentences + bad_sentences
labels = [1]*200 + [0]*200

vectorizer = TfidfVectorizer(ngram_range=(1,3))
X = vectorizer.fit_transform(data)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

joblib.dump(model, "grammar_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved!")

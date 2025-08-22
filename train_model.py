import requests
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def fetch_sentences(limit=1000):
    url = f"https://tatoeba.org/eng/api_v0/search?from=rus&orphans=no&query=&limit={limit}"
    r = requests.get(url)
    data = r.json()
    sentences = [s['text'] for s in data.get('results', []) if s.get('text')]
    return sentences

good_sentences = fetch_sentences(1000)

def make_noisy(sentence, p_delete=0.4, p_swap=0.3, p_char=0.3):
    words = sentence.split()
    words = [w for w in words if random.random() > p_delete]
    for i in range(len(words)-1):
        if random.random() < p_swap:
            words[i], words[i+1] = words[i+1], words[i]
    noisy = []
    for w in words:
        if random.random() < p_char and len(w) > 1:
            pos = random.randint(0, len(w)-1)
            w = w[:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + w[pos+1:]
        noisy.append(w)
    return ' '.join(noisy)

bad_sentences = [make_noisy(s) for s in good_sentences]

data = good_sentences + bad_sentences
labels = [1]*len(good_sentences) + [0]*len(bad_sentences)

vectorizer = TfidfVectorizer(ngram_range=(1,3))
X = vectorizer.fit_transform(data)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

joblib.dump(model, "grammar_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved!")

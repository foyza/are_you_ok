import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

X = np.array([[5], [10], [50], [100], [150], [200]])
y = np.array([0, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Модель обучена и сохранена как model.pkl")

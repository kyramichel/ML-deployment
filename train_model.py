import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

# Train a simple pipeline
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=200)
)
clf.fit(X, y)

# Persist the model
joblib.dump(clf, "model.pkl")
print("Model saved to model.pkl")

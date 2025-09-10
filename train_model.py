import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------------------
# Load datasets
# ---------------------------
dataset = pd.read_csv("dataset.csv", encoding="utf-8")
disease_description = pd.read_csv("disease_description.csv", encoding="utf-8")
disease_precaution = pd.read_csv("disease_precaution.csv", encoding="latin1")
symptom_severity = pd.read_csv("symptom_severity.csv", encoding="utf-8")
Training = pd.read_csv("Training.csv", encoding="utf-8")
Testing = pd.read_csv("Testing.csv", encoding="utf-8")

# ---------------------------
# Prepare training data
# ---------------------------
symptom_cols = [col for col in Training.columns if "Symptom_" in col]
Training["Symptoms_text"] = Training[symptom_cols].fillna("").agg(" ".join, axis=1)

X = Training["Symptoms_text"]
y = Training["Disease"]

# ---------------------------
# TF-IDF + Logistic Regression
# ---------------------------
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# ---------------------------
# Save model + vectorizer
# ---------------------------
with open("symptom_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained successfully!")
print("Test Accuracy:", model.score(X_test, y_test))

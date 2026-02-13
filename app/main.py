from fastapi import FastAPI
import joblib
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "tfidf_logistic.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}

@app.post("/predict")
def predict_sentiment(review: str):
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return {"prediction": sentiment}

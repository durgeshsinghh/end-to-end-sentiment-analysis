import streamlit as st
import joblib

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬")

st.title("🎬 IMDB Sentiment Analysis")

# Load vectorizer and model
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/tfidf_logistic.pkl")

review = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        # Transform input
        transformed_review = vectorizer.transform([review])

        # Predict
        prediction = model.predict(transformed_review)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"

        st.success(f"Prediction: {sentiment}")

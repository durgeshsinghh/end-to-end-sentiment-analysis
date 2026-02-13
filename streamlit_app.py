import streamlit as st
import requests

st.title("🎬 IMDB Sentiment Analysis")

review = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        params={"review": review}
    )

    result = response.json()

    st.success(f"Prediction: {result['prediction']}")

    

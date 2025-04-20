import streamlit as st
import joblib
import numpy as np

# Load the saved model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Hate Speech Detector", layout="centered")

st.title("🧠 Hate Speech Detection App")
st.write("This app detects whether a comment contains **hate speech**, **offensive language**, or **neither**.")

# User input
user_input = st.text_area("✍️ Enter your comment here:", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a valid comment.")
    else:
        # Transform input using the loaded vectorizer
        transformed_input = vectorizer.transform([user_input]).toarray()

        # Get prediction
        prediction = model.predict(transformed_input)[0]

        # Display result
        st.markdown("## 🧾 Prediction:")
        if prediction == "hate speech":
            st.error("🟥 Category: Hate Speech")
            st.markdown("*Hate is a heavy burden to bear, let kindness be your foundation.*")
        elif prediction == "offensive language":
            st.warning("🟧 Category: Offensive Language")
            st.markdown("*Your words can either build up or tear down — make them count.*")
        elif prediction == "neither hate nor offensive":
            st.success("🟩 Category: Neither Hate Nor Offensive")
            st.markdown("*Good job! Spread kindness over hate or offensive messages.*")
        else:
            st.info("🔵 Unknown Category")

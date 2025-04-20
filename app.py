import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Page config
st.set_page_config(page_title="Hate Speech Detector", layout="centered", page_icon="🧠")

# Header section with image/logo (optional)
st.markdown("""
    <style>
        .title-text {
            font-size: 36px;
            font-weight: bold;
            color: #4B8BBE;
            text-align: center;
        }
        .subtitle-text {
            font-size: 18px;
            color: #666;
            text-align: center;
            margin-top: -10px;
        }
        .footer {
            margin-top: 3rem;
            font-size: 13px;
            text-align: center;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-text">🧠 Hate Speech Detection App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Identify hate speech, offensive language, or positive content in your comments</p>', unsafe_allow_html=True)

# Divider
st.markdown("---")

# Input area
st.markdown("### ✍️ Enter your comment below:")
user_input = st.text_area("", height=150, placeholder="Type your comment here...")

if st.button("🚀 Analyze Now"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a valid comment to analyze.")
    else:
        transformed_input = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(transformed_input)[0]

        st.markdown("## 🧾 Result")
        if prediction == "hate speech":
            st.error("🟥 Category: Hate Speech")
            st.markdown(
                """
                > *“You either die a hero, or you live long enough to see yourself become the villain.” — The Dark Knight*
                
                💡 Let’s not be the villain in someone else’s story. Spread peace. ✌️
                """
            )

        elif prediction == "offensive language":
            st.warning("🟧 Category: Offensive Language")
            st.markdown(
                """
                > *“With great power comes great responsibility.” — Spider-Man*

                🗣️ Use your words wisely — they carry weight and power.
                """
            )

        elif prediction == "neither hate nor offensive":
            st.success("🟩 Category: Neither Hate Nor Offensive")
            st.markdown(
                """
                > *“After all, tomorrow is another day.” — Gone with the Wind*

                🌟 Your words are kind. Keep spreading positivity!
                """
            )

        else:
            st.info("🔵 Unknown Category")
            st.markdown("This comment couldn't be classified. Please try again.")

# Footer
st.markdown("<div class='footer'>Created by AYUSH SAINI ,  using Streamlit</div>", unsafe_allow_html=True)

import pickle
import re
import streamlit as st
from nltk.corpus import stopwords
import os

import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("background.jpg")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "sentiment_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
st.title("Tweet Mood Detector 🐦")
st.write("Type a tweet below to analyze its sentiment.")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#","", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

tweet = st.text_area(
    "Enter your tweet:",
    placeholder="I love this new phone!",
    height=120
)

if st.button("Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(tweet)
        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = max(proba) * 100

        st.subheader("Result")
        if confidence<65:
            st.info(f"This tweet has mixed sentiments.Confidence:{confidence:.2f}%")
        elif pred == "positive":
            st.success(f"this tweet shows positive sentiment!\n\n")
        else:
            st.error(f"This tweet shows negative sentiment.Be kinder!\n")


import streamlit as st
import pickle
import re
import os
import base64

STOPWORDS = set("""
a about above after again against all am an and any are arent as at
be because been before being below between both but by
can cant cannot could couldnt did didnt do does doesnt doing dont down during
each few for from further
had hadnt has hasnt have havent having he hed hell hes her here heres hers herself him himself his how hows
i id ill im ive if in into is isnt it its itself
lets me more most mustnt my myself
no nor not of off on once only or other ought our ours ourselves out over own
same shant she shed shell shes should shouldnt so some such
than that thats the their theirs them themselves then there theres these they theyd theyll theyre theyve this those through to too
under until up very
was wasnt we wed well were weve were werent what whats when whens where wheres which while who whos whom why whys with wont would wouldnt
you youd youll youre youve your yours yourself yourselves
""".split())



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
    text = " ".join([w for w in text.split() if w not in STOPWORDS])
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


import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf.pkl')

# NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return ' '.join(words)

# UI
st.title("NLP Text Classification")

user_input = st.text_area("Enter text")

if st.button("Predict"):

    cleaned = clean_text(user_input)

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)

    st.success(f"Predicted Class: {prediction[0]}")
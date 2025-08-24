import streamlit as st
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Stemmer
ps = PorterStemmer()

# Function to preprocess text
def text_transform(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric words
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Apply stemming

    text = y[:]
    return " ".join(y)

# Load the model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Classifier")
input_sms = st.text_area("Enter the message")

if st.button("Predict"):  # Only process if user inputs something
    transformed_sms = text_transform(input_sms)
    vector_input = tfidf.transform([transformed_sms])  # Correct method name
    result = model.predict(vector_input)[0]

    # Output result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Ham")


import streamlit as st
import pickle
import string
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ------------------------------
# Setup persistent NLTK data folder
# ------------------------------
nltk_data_path = os.environ.get("NLTK_DATA", "/opt/render/project/src/nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)

# Initialize Stemmer
ps = PorterStemmer()

# ------------------------------
# Function to preprocess text
# ------------------------------
def text_transform(text):
    text = text.lower()
    text = word_tokenize(text)

    y = [i for i in text if i.isalnum()]  # Keep only alphanumeric words

    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    stemmed_text = [ps.stem(i) for i in text]  # Apply stemming

    return " ".join(stemmed_text)

# ------------------------------
# Load the model and vectorizer
# ------------------------------
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Email/SMS Classifier")
input_sms = st.text_area("Enter the message")

if st.button("Predict") and input_sms.strip():
    transformed_sms = text_transform(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    # Output result
    st.header("Spam" if result == 1 else "Ham")

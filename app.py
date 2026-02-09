import streamlit as st
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# load model + vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    cv = pickle.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    r = re.sub('[^a-zA-Z]', ' ', text)
    r = r.lower()
    r = r.split()
    r = [word for word in r if word not in stop_words]
    r = [lemmatizer.lemmatize(word) for word in r]
    return ' '.join(r)

st.title("Spam Classifier (Logistic Regression)")

user_input = st.text_area("Enter a message to classify:")

if st.button("Predict"):
    cleaned = preprocess(user_input)
    vectorized = cv.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        st.error("This looks like SPAM")
    else:
        st.success("This looks NOT SPAM")
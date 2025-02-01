import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


ps = PorterStemmer()

model=pickle.load(open('mnb.pkl', 'rb'))
tf=pickle.load(open('vectorizer.pkl', 'rb'))


def transform_text(text):
    text = text.lower()
    text = text.split()

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title('EMAIL/SMS SPAM CLASSIFIER')

#get input
input_msg=st.text_area('Enter the message')
if st.button('Predict'):
    # preprocessing
    transformed_msg =transform_text(input_msg)
    # vectorize the input by tfidf vectorizer
    vector_input = tf.transform([transformed_msg])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header('Spam ❌')
    else:
        st.header('Not Spam 🎉')








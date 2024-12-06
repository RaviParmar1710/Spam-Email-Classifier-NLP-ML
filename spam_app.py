import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load Model and Vectorizer
with open("spam_class.pkl","rb") as model_file:
    model = pickle.load(model_file)

with open("count_v.pkl","rb") as countvectorizer_file:
    count_v = pickle.load(countvectorizer_file)


# Streamlit UI
st.title("Spam Email Classifier")
st.write("A simple app to classify emails as **Spam** or **Ham**.")


# Input Text Box
email_text = st.text_area("Enter the email text:")
if st.button("Classify"):
    if email_text.strip() =="":
        st.warning("Please enter an email to classify.")
    else:
        # Preprocess and Predict
        email_vectorizer = count_v.transform([email_text])
        prediction   = model.predict(email_vectorizer)
        result = "Spam" if prediction[0] ==1 else "Ham"
        st.success(f"The email is Classified As: **{result}**.")
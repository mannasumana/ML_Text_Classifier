import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Text Classification Web App")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip() != "":
        # Convert to dense array for SVC
        transformed = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(transformed)[0]
        st.success(f"Prediction: {prediction}")
    else:
        st.warning("Please enter some text")
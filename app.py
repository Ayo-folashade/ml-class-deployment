import streamlit as st
import joblib

# Load the model and the vectorizer
model = joblib.load('text_classification_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit interface
st.title('NLP Text Classification App')

# Add image
st.image('sentiment.jpg')

# Collect user input
user_input = st.text_area("Enter the text for prediction")

# Button to trigger prediction
if st.button('Predict'):
    if user_input:
        # Transform the user input using the vectorizer
        user_input_vector = vectorizer.transform([user_input])

        # Make prediction for the user input
        prediction = model.predict(user_input_vector)

        # Output the prediction
        if prediction == -1:
            st.write('Prediction: Negative')
        elif prediction == 0:
            st.write('Prediction: Neutral')
        else:
            st.write('Prediction: Positive')
    else:
        st.write('Please enter text for classification.')


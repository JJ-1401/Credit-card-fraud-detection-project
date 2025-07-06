import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random

# Load the trained model, scaler, and feature dataset
model = joblib.load('svm_model_balanced.pkl')
scaler = joblib.load('scaler.pkl')
feature_data = pd.read_csv('feature_data.csv')

st.title('💳 Credit Card Fraud Detection Web App')

st.write('Enter transaction amount to predict whether it is Fraudulent or Not Fraudulent.')

# User input for the transaction amount
amount = st.number_input('Transaction Amount', min_value=0.0, max_value=100000000.0, value=100.0)

if st.button('Predict'):
    # Select a random row from the dataset for the 29 hidden features
    random_row = feature_data.sample(n=1, random_state=random.randint(0, 100000000))
    
    # Get all features except 'Amount'
    input_features = random_row.values.flatten().tolist()
    
    # Find index of the 'Amount' column (adjust this if 'Amount' is not the last column)
    amount_index = feature_data.columns.get_loc('Amount')
    
    # Replace the 'Amount' value in the random row with user input (scaled)
    scaled_amount = scaler.transform(np.array([[amount]]))[0][0]
    input_features[amount_index] = scaled_amount

    # Convert to numpy array
    input_array = np.array([input_features])

    # Make prediction
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.error('⚠️ Fraudulent Transaction Detected!')
    else:
        st.success('✅ Transaction is Not Fraudulent.')

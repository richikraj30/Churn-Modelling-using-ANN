import streamlit as st
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# Load the trained model

model = tf.keras.models.load_model('artifacts/ann_model.h5')

# Load the encoder and scaler

with open('artifacts/label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('artifacts/onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Streamlit app
st.title('Predicting Customer Churn')

# User input

gender = st.selectbox('Gender', label_encoder_gender.classes_)
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
age = st.slider('Age', 18, 95)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])
num_of_products = st.slider('Number of Products', 1, 4)

# Prepare the input data

input_data = pd.DataFrame(
    {
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
    }
)

# One-hot encode the geography
geo_ecoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_ecoded_df = pd.DataFrame(geo_ecoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine the one-hot encoded geography with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_ecoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make predictions
predictions = model.predict(input_data_scaled)
predictions_probability = predictions[0][0]

# Display the results
if predictions_probability > 0.5:
    st.write(f'The customer is predicted to leave the bank with a probability of {predictions_probability:.2%}')
else:
    st.write(f'The customer is predicted to stay with the bank with a probability of {1-predictions_probability:.2%}')
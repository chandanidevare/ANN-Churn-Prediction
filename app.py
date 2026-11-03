import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder



# Load the trained model:
model=tf.keras.models.load_model('model.h5')




# Load the encoders & scaler:
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)




# Streamlit App:
st.title("Customer Churn Prediction Using ANN Model")




# User Input:
# User Input
credit_score = st.number_input("Credit Score", 300, 900)

geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.selectbox("Age", list(range(18, 101)))

tenure = st.selectbox("Tenure", list(range(0, 11)))

balance = st.number_input("Balance", min_value=0.0)

num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])

has_cr_card = st.selectbox("Has Credit Card", [0, 1])

is_active_member = st.selectbox("Is Active Member", [0, 1])

estimated_salary = st.number_input("Estimated Salary", min_value=0.0)

#model.save('model.



# Prepare the input data:
input_data = pd.DataFrame({
'CreditScore':[credit_score],
'Gender':[label_encoder_gender.transform([gender])[0]], 
'Age':[age],
'Tenure':[tenure],
'Balance':[balance],
'NumOfProducts':[num_of_products],
'HasCrCard':[has_cr_card], 
'IsActiveMember':[is_active_member],
 'EstimatedSalary':[estimated_salary],
        # 'Geography':[onehot_encoder_geography.transform([[geography]])[0][0]]
})




# one hot encode the geography:
geo_encoded= onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))



# Combine onehot encoded columns with input data:
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df],axis=1)




# Scale the input data:
input_data_scaled = scaler.transform(input_data)


# Prediction the churn:
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]


st.write(f"Chrun Probability: {prediction_probability:.2f}")

if prediction_probability > 0.5:
    st.write("The customer is likely to chrun.")
else:
    st.write("The customer is not likely to chrun.")
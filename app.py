
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📈",
    layout="wide"
)


# CUSTOM CSS



st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #0B1530;
    color: #FFFFFF;
}

/* PAGE LAYOUT */
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* MAIN HEADING */
h1 {
    color: #FFFFFF !important;
    font-size: 36px !important;
    font-weight: 800 !important;
}

/* SUBHEADINGS */
h2, h3 {
    color: #E0F2FE !important;
}

/* NORMAL TEXT */
.stApp p {
    color: #CBD5E1;
}

/* CUSTOMER DETAILS FORM - DARK BLUE */
div[data-testid="stForm"] {
    background-color: #162544 !important;
    padding: 28px !important;
    border-radius: 16px !important;
    border: 1px solid #334B70 !important;
    box-shadow: 0 5px 20px rgba(0,0,0,0.25);
}

/* FORM HEADING */
div[data-testid="stForm"] h2,
div[data-testid="stForm"] h3,
div[data-testid="stForm"] p {
    color: #FFFFFF !important;
}

/* INPUT LABELS */
div[data-testid="stForm"] label {
    color: #E2E8F0 !important;
    font-weight: 600 !important;
}

/* NUMBER INPUT */
div[data-testid="stForm"] input {
    background-color: #FFFFFF !important;
    color: #172554 !important;
    border: 1px solid #94A3B8 !important;
    border-radius: 9px !important;
}

/* INPUT FOCUS */
div[data-testid="stForm"] input:focus {
    border: 2px solid #06B6D4 !important;
    box-shadow: 0 0 0 2px rgba(6,182,212,0.15) !important;
}

/* DROPDOWN */
div[data-testid="stForm"] div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border-radius: 9px !important;
}

/* DROPDOWN TEXT */
div[data-testid="stForm"] div[data-baseweb="select"] * {
    color: #172554 !important;
}

/* DROPDOWN OPTIONS */
div[data-baseweb="popover"],
div[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
}

div[role="option"] {
    color: #172554 !important;
}

/* PREDICT BUTTON */
.stFormSubmitButton button {
    background-color: #0891B2 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    min-height: 45px;
    transition: 0.3s;
}

/* BUTTON HOVER */
.stFormSubmitButton button:hover {
    background-color: #0E7490 !important;
    color: #FFFFFF !important;
}

/* RESULT ALERT */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* DIVIDER */
hr {
    border-color: #334155 !important;
}

/* RESPONSIVE DESIGN */
@media (max-width: 768px) {
    h1 {
        font-size: 27px !important;
    }

    div[data-testid="stForm"] {
        padding: 16px !important;
    }
}

</style>
""", unsafe_allow_html=True)




# LOAD MODEL AND ENCODERS
@st.cache_resource
def load_resource():

    model = tf.keras.models.load_model('model.h5')

    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_resource()


# STREAMLIT APP TITLE
#st.title(":material/neurology: Customer Churn Prediction")
st.title(" 📈 Customer Churn Prediction Using ANN Model")

st.markdown(
    "<p style='text-align:center; color:#cbd5e1;'>"
    "Predict customer behaviour using Artificial Neural Networks (ANN)."
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# USER INPUT
st.subheader("👤 Enter Customer Details")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(" 🤝 Personal Details")

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )

        geography = st.selectbox(
            "Geography",
            onehot_encoder_geo.categories_[0]
        )

        gender = st.selectbox(
            "Gender",
            label_encoder_gender.classes_
        )

        age = st.selectbox(
            "Age",
            list(range(18, 101)),
            index=22
        )

        tenure = st.selectbox(
            "Tenure (Years)",
            list(range(0, 11)),
            index=5
        )

    with col2:

        st.subheader(" 🏦 Financial & Account Details")

        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            value=50000.0
        )

        num_of_products = st.selectbox(
            "Number of Products",
            [1, 2, 3, 4]
        )

        has_cr_card = st.selectbox(
            "Has Credit Card",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        is_active_member = st.selectbox(
            "Is Active Member",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn"
    )


# PREDICTION
if submitted:

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # ONE HOT ENCODE GEOGRAPHY
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # COMBINE ENCODED DATA
    input_data = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    # SCALE INPUT DATA
    input_data_scaled = scaler.transform(input_data)

    # PREDICTION
    prediction = model.predict(input_data_scaled)

    prediction_probability = prediction[0][0]

    # DISPLAY RESULT
    st.divider()
    st.subheader("🎯 Prediction Result")

    st.metric(
        "Churn Probability",
        f"{prediction_probability * 100:.2f}%"
    )

    st.progress(float(prediction_probability))

    if prediction_probability > 0.5:
        st.error("✅ The customer is likely to churn.")
    else:
        st.success("⚠️ The customer is not likely to churn.")


# FOOTER
st.divider()

st.markdown(
    "<p style='text-align:center; color:#94a3b8;'>"
    "Customer Churn Prediction | Artificial Neural Network (ANN)| " \
    "Python | Streamlit |"
    "</p>",
    unsafe_allow_html=True
)

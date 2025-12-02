# apps.py
import streamlit as st
import pandas as pd
import joblib

# Load Model + Encoders
@st.cache_resource
def load_model():
    model = joblib.load("kmeans_model.joblib")   # KMeans model
    encoders = joblib.load("encoders.joblib")    # label encoders
    return model, encoders

model, encoders = load_model()

# Define features
features = ['sector', 'referral_source', 'account_type', 'country']

st.title("Merchant Activation Cluster Prediction (KMeans)")
st.write("Enter merchant details below to determine activation likelihood cluster.")

# Sidebar Inputs
st.sidebar.header("Merchant Information")
user_input = {}

for feature in features:
    options = encoders[feature].classes_.tolist()
    user_input[feature] = st.sidebar.selectbox(f"{feature}", options)

# Encode user input
encoded_input = {}
for feature in features:
    encoded_input[feature] = encoders[feature].transform([user_input[feature]])[0]

# Convert to DataFrame
input_df = pd.DataFrame([encoded_input])
input_df = input_df[features]  # enforce column order

# KMeans Prediction
cluster = model.predict(input_df)[0]

# Cluster → Activation Probability Mapping
cluster_map = {
    0: 0.233, #highest conversion rate
    1: 0.142, # lowest converstion rate
    2: 0.178,
    3: 0.173
}

activation_cluster = cluster_map.get(cluster, 0.0)

# Display Results
st.subheader("Cluster Result")
st.write(f"Merchant belongs to **Cluster {cluster}**")

st.subheader("Activation Cluster")
st.write(f"Activation rate for the cluster: **{activation_cluster:.2f}**")

st.progress(float(activation_cluster))

if activation_cluster >= 0.23:
    st.success(" Strong chance of activation — This merchant profile is highly likely to activate.")
elif 0.17 <= activation_cluster < 0.23:
    st.warning(" Moderate chance — Merchant may activate but may need light assistance.")
else:
    st.error("Low activation likelihood — This profile historically struggles to activate.")

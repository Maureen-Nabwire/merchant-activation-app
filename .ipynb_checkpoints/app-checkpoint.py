import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load preprocessor + models
# -----------------------------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.joblib")

    models = {
       # "Decision Tree": joblib.load("decision_tree_model.joblib"),
        "Gradient Boosting": joblib.load("gradient_boosting_model.joblib"),
       # "Best Gradient Boosting (Tuned)": joblib.load("best_gradient_boosting.joblib")
    }

    return preprocessor, models

preprocessor, models = load_artifacts()

# -----------------------------
# Extract dropdown options from OneHotEncoder
# -----------------------------
def extract_feature_categories(preprocessor, raw_feature_names):
    """
    Reads OneHotEncoder categories_ directly from the preprocessor.
    Returns dictionary: {feature: [list of categories]}
    """
    ohe = preprocessor.named_transformers_['cat']  # OneHotEncoder
    categories = ohe.categories_

    feature_options = {}
    for feature_name, cats in zip(raw_feature_names, categories):
        feature_options[feature_name] = list(cats)

    return feature_options

# Raw input features (the same as in training notebook)
features = ['sector', 'referral_source', 'account_type', 'country']

# Extract options from the fitted OHE
dropdown_options = extract_feature_categories(preprocessor, features)


# Streamlit UI
st.title(" Merchant Activation Prediction App")
st.write("Use the sidebar to enter merchant details and choose a machine learning model.")

# Sidebar
st.sidebar.header("Merchant Input Data")

user_input = {}
for feature in features:
    user_input[feature] = st.sidebar.selectbox(
        f"{feature.replace('_', ' ').title()}",
        dropdown_options[feature]
    )

# Model selection
model_choice = st.sidebar.selectbox(
    "Choose Prediction Model",
    list(models.keys())
)

model = models[model_choice]

# Prepare input for prediction
input_df = pd.DataFrame([user_input])

# Transform using the exact training preprocessor
X_transformed = preprocessor.transform(input_df)

# Prediction
prediction = model.predict(X_transformed)[0]
probability = model.predict_proba(X_transformed)[0][1]

# Display Results
st.subheader("Prediction Result")

if prediction == 1:
    st.success(f"Merchant is LIKELY to activate (probability: {probability:.2f})")
else:
    st.error(f"Merchant is NOT likely to activate (probability: {probability:.2f})")

st.subheader(" Activation Probability")
st.write(f"**{probability:.4f}**")

st.progress(float(probability))

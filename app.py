import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model AND feature names
model, feature_names = pickle.load(open("model/model.pkl", "rb"))

st.title("🛡 SocialGuard - AI Fake Profile Detection")

st.write("Enter profile details:")

user_input = {}

for feature in feature_names:
    if feature in ["profile_pic", "is_private", "is_verified"]:
        user_input[feature] = st.selectbox(f"{feature}", [0, 1])
    else:
        user_input[feature] = st.number_input(f"{feature}", min_value=0)

if st.button("Check Profile"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠ This Profile is Likely FAKE")
    else:
        st.success("✅ This Profile is Likely REAL")

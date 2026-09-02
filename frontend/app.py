import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="Home Credit Risk Dashboard", page_icon="💳", layout="centered")

@st.cache_data
def get_data():
    response = requests.get(f"{BACKEND_URL}/applicants/")
    return response.json()

applicant_ids = get_data()


st.title("Home Credit Default Risk Dashboard")
st.caption("Select a sample applicant, then adjust their declarable details to see how risk changes. "
           "Bureau and credit-history data stay fixed to the sampled applicant's real records.")

st.divider()

sk_id = st.selectbox("Select Applicant ID", applicant_ids)
response = requests.get(f"{BACKEND_URL}/applicants/{sk_id}")
row = response.json()

st.subheader("Application Details")

col1, col2 = st.columns(2)
with col1:
    income = st.number_input(
        "Annual Income", value=float(row["Income"]),
        min_value=0.0, max_value=10_000_000.0
    )
    annuity = st.number_input(
        "Annuity", value=float(row["Annuity"]),
        min_value=0.0, max_value=1_000_000.0
    )
    family_members = st.slider(
        "Family Members", 1, 10, value=int(row["Family_members"])
    )
with col2:
    credit = st.number_input(
        "Credit Amount", value=float(row["Credit"]),
        min_value=0.0, max_value=10_000_000.0
    )
    good_price = st.number_input(
        "Goods Price", value=float(row["Good_price"]),
        min_value=0.0, max_value=10_000_000.0
    )
    days_employed = st.number_input(
        "Days Employed (negative = currently employed)",
        value=float(row["Days_employed"]),
        min_value=-20_000.0, max_value=0.0
    )

st.divider()

if st.button("Predict Risk", type="primary"):
    with st.spinner("Calculating risk..."):
        payload = {
            "Id": sk_id,
            "Income": income,
            "Credit": credit,
            "Annuity": annuity,
            "Good_price": good_price,
            "Family_members": family_members,
            "Days_employed": days_employed
        }
        try:
            response = requests.post(f"{BACKEND_URL}/predict/", json=payload)
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the prediction server. Is it running?")
            st.stop()

        if response.status_code != 200:
            st.error(f"Prediction failed: {response.json().get('detail', 'Unknown error')}")
            st.stop()

        result = response.json()
        prob = result["probability"]
        risk_band = result["risk_band"]
        shap_data = result["shap_values"]

    st.toast("Prediction complete", icon="✅")

    st.subheader("Result")
    m1, m2 = st.columns(2)
    m1.metric("Default Probability", f"{prob:.2%}")
    m2.metric("Risk Band", risk_band)

    if risk_band == "Low":
        st.success(f"Low Risk — Default Probability: {prob:.2%}")
    elif risk_band == "Medium":
        st.warning(f"Medium Risk — Default Probability: {prob:.2%}")
    else:
        st.error(f"High Risk — Default Probability: {prob:.2%}")

    st.divider()
    st.subheader("Why this prediction?")
    st.caption("Top factors driving this applicant's risk score. "
               "Red bars push risk higher, green bars push risk lower.")

    shap_df = pd.DataFrame(shap_data)
    top_features = shap_df.sort_values("impact")

    colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_features["impact"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top_features["feature"], top_features["impact"], color=colors)
    ax.set_xlabel("SHAP value (impact on default probability)")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()

    st.pyplot(fig)

import pickle
import pandas as pd

import os
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
features_path = os.path.join(BASE_DIR, "features.pkl")
parquet_path = os.path.join(BASE_DIR, "test_merged.parquet")


def load_model():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_features():
    with open(features_path, "rb") as f:
        selected_features = pickle.load(f)
    return selected_features


def load_data():
    df = pd.read_parquet(parquet_path)
    return df


def get_application_data(df, SK_ID_CURR):
    row = df[df["SK_ID_CURR"] == SK_ID_CURR].copy()
    return row


def recompute(row):
    row["INCOME_PER_PERSON"] = row["AMT_INCOME_TOTAL"] / row["CNT_FAM_MEMBERS"].replace(0, 1)
    row["EMPLOYED_TO_BIRTH"] = row["DAYS_EMPLOYED"] / row["DAYS_BIRTH"]
    row["CREDIT_INCOME_RATIO"] = row["AMT_CREDIT"] / row["AMT_INCOME_TOTAL"]
    row["ANNUITY_INCOME_RATIO"] = row["AMT_ANNUITY"] / row["AMT_INCOME_TOTAL"]
    row["CREDIT_TERM"] = row["AMT_ANNUITY"] / row["AMT_CREDIT"]
    row["GOODS_CREDIT_RATIO"] = row["AMT_GOODS_PRICE"] / row["AMT_CREDIT"]
    row["EMPLOYED_YEARS"] = row["DAYS_EMPLOYED"].clip(upper=0) / -365
    return row


def predict(model, row, selected_features):
    row = row.reindex(columns=selected_features)
    prob = model.predict_proba(row)[:, 1]
    return prob[0]

def explain_prediction(model, row, selected_features):
    row = row.reindex(columns=selected_features)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)
    return shap_values, explainer.expected_value
if __name__=="__main__":
    model = load_model()
    selected_features = load_features()
    df = load_data()
    row = get_application_data(df, 100001)
    row = recompute(row)
    prob = predict(model, row, selected_features)
    print("Default probability:", prob)
    shap_values, base_value = explain_prediction(model, row, selected_features)
    print("Base value:", base_value)
    print("SHAP values shape:", shap_values.shape)
    print("SHAP values:", shap_values)

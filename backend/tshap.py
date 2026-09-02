import os
import pickle
import shap
from inference import load_model, load_features, load_data, get_application_data, recompute

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run():
    model = load_model()
    selected_features = load_features()
    df = load_data()

    row = get_application_data(df, 100001)   # same test applicant you used before
    row = recompute(row)
    row = row.reindex(columns=selected_features)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)

    print("Prediction probability:", model.predict_proba(row)[:, 1][0])
    print("\nSHAP values for this applicant:")
    for feat, val in zip(selected_features, shap_values[0]):
        print(f"{feat}: {val:.4f}")

if __name__ == "__main__":
    run()

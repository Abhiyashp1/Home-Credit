import os
import pickle
import pandas as pd
from merge import load_and_merge
from data_pre import clean_data, feature_engineering

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "backend")

features_path = os.path.join(BACKEND_DIR, "features.pkl")
output_path   = os.path.join(BACKEND_DIR, "test_merged.parquet")
impute_stats_path = os.path.join(BACKEND_DIR, "impute_stats.pkl")

def prep_test_data():
    with open(features_path, "rb") as f:
        selected_features = pickle.load(f)
    with open(impute_stats_path,"rb")as f:
         impute_stats = pickle.load(f)
    df = load_and_merge("application_test.csv")
    df = feature_engineering(df)
    df = clean_data(df,stats=impute_stats)
    if "SK_ID_CURR" not in selected_features:
        selected_features = ["SK_ID_CURR"] + selected_features
    df = df.reindex(columns=selected_features, fill_value=0)
    df.to_parquet(output_path, index=False)
    print("saved:", df.shape)

if __name__ == "__main__":
    prep_test_data()

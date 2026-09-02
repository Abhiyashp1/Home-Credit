import os
from xgboost import XGBClassifier
import pandas as pd
from data_pre import clean_data, split_data, feature_selection, feature_engineering
from sklearn.metrics import roc_auc_score
import optuna
import pickle
from sklearn.model_selection import train_test_split
from merge import load_and_merge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path    = os.path.join(BASE_DIR, "model.pkl")
features_path = os.path.join(BASE_DIR, "features.pkl")

def train_model():
    df = load_and_merge()
    df = feature_engineering(df)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, selected_features = feature_selection(X_train, y_train, X_test)

    # handle class imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos
    print(f"scale_pos_weight set to: {scale:.2f}")

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 900),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma":             trial.suggest_float("gamma", 0, 1),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 2),
        }

        model = XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
            scale_pos_weight=scale
        )
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred)

    # run optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print(f"Best ROC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # retrain with best params
    best_model = XGBClassifier(
        **study.best_params,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale
    )
    best_model.fit(X_train, y_train)

    # save model and features
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(features_path, "wb") as f:
        pickle.dump(selected_features, f)
    print("Model and features saved successfully")

    return best_model

if __name__ == "__main__":
    train_model()

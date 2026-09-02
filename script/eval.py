import os
import pickle
from merge import load_and_merge
from data_pre import clean_data, split_data, feature_engineering
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(BASE_DIR)
BACKEND_DIR = os.path.join(DATA_DIR, "backend")

data_path     = os.path.join(DATA_DIR, "application_train.csv")
model_path    = os.path.join(BACKEND_DIR, "model.pkl")
features_path = os.path.join(BACKEND_DIR, "features.pkl")

def evaluate_model():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path, "rb") as f:
        selected_features = pickle.load(f)

    df = load_and_merge()
    df=feature_engineering(df)
    df,_= clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # use saved features — no rerunning feature selection
    X_test = X_test[selected_features]

    y_pred_proba  = model.predict_proba(X_test)[:, 1]
    threshold     = 0.6
    y_pred_labels = (y_pred_proba >= threshold).astype(int)

    print("ROC score:", roc_auc_score(y_test, y_pred_proba))
    print("Classification report:\n", classification_report(y_test, y_pred_labels))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_labels))

if __name__ == "__main__":
    evaluate_model()

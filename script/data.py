import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(BASE_DIR)

data = os.path.join(DATA_DIR, "application_train.csv")

def feature_engineering(df):
    # Flag anomalous DAYS_EMPLOYED (365243 = placeholder for unemployed/retired)
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    # Credit ratios
    df['CREDIT_INCOME_RATIO']  = df['AMT_CREDIT']      / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY']     / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM']          = df['AMT_ANNUITY']     / df['AMT_CREDIT']
    df['GOODS_CREDIT_RATIO']   = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']

    # Age and employment
    df['AGE_YEARS']         = df['DAYS_BIRTH']    / -365
    df['EMPLOYED_YEARS']    = df['DAYS_EMPLOYED'].clip(upper=0) / -365
    df['EMPLOYED_TO_BIRTH'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # Income per family member
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, 1)

    # DPD ratio features — strongest predictors of default
    df['BUREAU_DEBT_CREDIT_RATIO'] = df['BUREAU_AMT_DEBT_MEAN']  / (df['BUREAU_AMT_CREDIT_MEAN'].abs()  + 1)
    df['INST_PAYMENT_RATIO']       = df['INST_AMT_PAYMENT_MEAN'] / (df['INST_AMT_DIFF_MEAN'].abs()       + 1)
    df['CC_DPD_RATIO']             = df['CC_SK_DPD_MEAN']        / (df['CC_AMT_BALANCE_MEAN'].abs()      + 1)
    df['POS_DPD_RATIO']            = df['POS_SK_DPD_MEAN']       / (df['POS_CNT_INSTALMENT_MEAN']        + 1)

    return df

def clean_data(df, remove_outliers=False, fix_skew=False, stats=None):
    fit_mode = stats is None
    if fit_mode:
        stats = {}

    # 1. Drop columns with too many missing values
    if fit_mode:
        missing = df.isnull().sum().sort_values(ascending=False)
        missing_percent = (missing / len(df)) * 100
        missing_df = pd.DataFrame({"missing count": missing, "missing %": missing_percent})
        cols_drop = missing_df[missing_df["missing %"] > 50].index
        stats["cols_drop"] = cols_drop
    else:
        cols_drop = stats["cols_drop"]
    df = df.drop(cols_drop, axis=1, errors="ignore")

    # 2. Fill missing values
    if fit_mode:
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = df.select_dtypes(include=["object"]).columns
        stats["num_medians"] = {}
        stats["cat_modes"] = {}
        for col in num_cols:
            median_val = df[col].median()
            stats["num_medians"][col] = median_val
            df[col] = df[col].fillna(median_val)
        for col in cat_cols:
            mode_val = df[col].mode()[0]
            stats["cat_modes"][col] = mode_val
            df[col] = df[col].fillna(mode_val)
    else:
        for col in stats["num_medians"]:
            if col in df.columns:
                df[col] = df[col].fillna(stats["num_medians"][col])
        for col in stats["cat_modes"]:
            if col in df.columns:
                df[col] = df[col].fillna(stats["cat_modes"][col])

    # 3. Remove outliers using IQR
    if remove_outliers:
        if fit_mode:
            num_cols = df.select_dtypes(include=["int64", "float64"]).columns
            num_cols = [col for col in num_cols if col != "TARGET"]
            stats["outlier_bounds"] = {}
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                stats["outlier_bounds"][col] = (lower, upper)
                df[col] = df[col].clip(lower=lower, upper=upper)
        else:
            for col, (lower, upper) in stats["outlier_bounds"].items():
                if col in df.columns:
                    df[col] = df[col].clip(lower=lower, upper=upper)

    # 4. Fix skew
    if fix_skew:
        if fit_mode:
            num_cols = df.select_dtypes(include=["int64", "float64"]).columns
            num_cols = [col for col in num_cols if col != "TARGET"]
            stats["skew_type"] = {}
            for col in num_cols:
                skew_val = df[col].skew()
                if skew_val > 1:
                    stats["skew_type"][col] = "right"
                    df[col] = np.log1p(df[col].clip(lower=0))
                elif skew_val < -1:
                    stats["skew_type"][col] = "left"
                    stats.setdefault("skew_max", {})[col] = df[col].max()
                    df[col] = np.log1p(df[col].max() - df[col])
        else:
            for col, kind in stats.get("skew_type", {}).items():
                if col not in df.columns:
                    continue
                if kind == "right":
                    df[col] = np.log1p(df[col].clip(lower=0))
                elif kind == "left":
                    max_val = stats["skew_max"][col]
                    df[col] = np.log1p(max_val - df[col])

    # 5. Encode categoricals
    if fit_mode:
        cat_cols = df.select_dtypes(include="object").columns
        binary_cols, multi_cols = [], []
        for col in cat_cols:
            if df[col].nunique() == 2:
                binary_cols.append(col)
            else:
                multi_cols.append(col)

        stats["label_encoders"] = {}
        for col in binary_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            stats["label_encoders"][col] = le

        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        stats["dummy_columns"] = df.columns.tolist()
    else:
        for col, le in stats["label_encoders"].items():
            if col in df.columns:
                df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        multi_cols = [c for c in df.select_dtypes(include="object").columns]
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        df = df.reindex(columns=stats["dummy_columns"], fill_value=0)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    if fit_mode:
        return df, stats
    else:
        return df

def split_data(df):
    X = df.drop(["TARGET","SK_ID_CURR"], axis=1)
    y = df["TARGET"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42, test_size=0.2
    )
    print("CLEANED AND SPLIT DATA")
    return X_train, X_test, y_train, y_test

def feature_selection(X_train, y_train, X_test, top_n=70):
    mi = mutual_info_classif(X_train, y_train)
    mi_scores = pd.Series(mi, index=X_train.columns)

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_scores = pd.Series(rf.feature_importances_, index=X_train.columns)

    feature_scores = pd.DataFrame({'MI': mi_scores, 'RF': rf_scores})
    feature_scores['MI_NORM'] = feature_scores['MI'] / feature_scores['MI'].max()
    feature_scores['RF_NORM'] = feature_scores['RF'] / feature_scores['RF'].max()
    feature_scores['FINAL'] = (feature_scores['MI_NORM'] + feature_scores['RF_NORM']) / 2
    feature_scores = feature_scores.sort_values(by='FINAL', ascending=False)

    selected_features = feature_scores.head(top_n).index.tolist()
    return X_train[selected_features], X_test[selected_features], selected_features

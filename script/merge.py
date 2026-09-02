import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src
DATA_DIR = os.path.dirname(BASE_DIR)                     # repo root

base_path = DATA_DIR

def load_and_merge(main_file="application_train.csv"):
    df = pd.read_csv(f"{base_path}\\{main_file}")
    print(f"Main table: {df.shape}")

    # bureau
    bureau = pd.read_csv(f"{base_path}\\bureau.csv")
    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT        = ("SK_ID_BUREAU",          "count"),
        BUREAU_AMT_CREDIT_SUM    = ("AMT_CREDIT_SUM",        "sum"),
        BUREAU_AMT_CREDIT_MEAN   = ("AMT_CREDIT_SUM",        "mean"),
        BUREAU_AMT_DEBT_MEAN     = ("AMT_CREDIT_SUM_DEBT",   "mean"),
        BUREAU_AMT_OVERDUE_MEAN  = ("AMT_CREDIT_SUM_OVERDUE","mean"),
        BUREAU_DAYS_CREDIT_MEAN  = ("DAYS_CREDIT",           "mean"),
        BUREAU_DAYS_CREDIT_MIN   = ("DAYS_CREDIT",           "min"),
        BUREAU_OVERDUE_MEAN      = ("CREDIT_DAY_OVERDUE",    "mean"),
        BUREAU_OVERDUE_MAX       = ("CREDIT_DAY_OVERDUE",    "max"),
    ).reset_index()
    df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")
    print(f"After bureau merge: {df.shape}")

    # previous applications
    prev = pd.read_csv(f"{base_path}\\previous_application.csv")
    prev_agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT           = ("SK_ID_PREV",    "count"),
        PREV_AMT_CREDIT_MEAN     = ("AMT_CREDIT",    "mean"),
        PREV_AMT_CREDIT_SUM      = ("AMT_CREDIT",    "sum"),
        PREV_AMT_ANNUITY_MEAN    = ("AMT_ANNUITY",   "mean"),
        PREV_AMT_APPLICATION_MEAN= ("AMT_APPLICATION","mean"),
        PREV_AMT_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT","mean"),
        PREV_DAYS_DECISION_MEAN  = ("DAYS_DECISION", "mean"),
        PREV_CNT_PAYMENT_MEAN    = ("CNT_PAYMENT",   "mean"),
    ).reset_index()
    df = df.merge(prev_agg, on="SK_ID_CURR", how="left")
    print(f"After previous_application merge: {df.shape}")

    # installments payments
    inst = pd.read_csv(f"{base_path}\\installments_payments.csv")
    inst["DAYS_LATE"]  = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["AMT_DIFF"]   = inst["AMT_INSTALMENT"]     - inst["AMT_PAYMENT"]
    inst_agg = inst.groupby("SK_ID_CURR").agg(
        INST_COUNT               = ("SK_ID_PREV",    "count"),
        INST_AMT_PAYMENT_SUM     = ("AMT_PAYMENT",   "sum"),
        INST_AMT_PAYMENT_MEAN    = ("AMT_PAYMENT",   "mean"),
        INST_DAYS_LATE_MEAN      = ("DAYS_LATE",     "mean"),
        INST_DAYS_LATE_MAX       = ("DAYS_LATE",     "max"),
        INST_AMT_DIFF_MEAN       = ("AMT_DIFF",      "mean"),
        INST_AMT_DIFF_MAX        = ("AMT_DIFF",      "max"),
    ).reset_index()
    df = df.merge(inst_agg, on="SK_ID_CURR", how="left")
    print(f"After installments merge: {df.shape}")

    # credit card balance
    cc = pd.read_csv(f"{base_path}\\credit_card_balance.csv")
    cc_agg = cc.groupby("SK_ID_CURR").agg(
        CC_COUNT                 = ("SK_ID_PREV",            "count"),
        CC_AMT_BALANCE_MEAN      = ("AMT_BALANCE",           "mean"),
        CC_AMT_BALANCE_MAX       = ("AMT_BALANCE",           "max"),
        CC_AMT_DRAWINGS_MEAN     = ("AMT_DRAWINGS_CURRENT",  "mean"),
        CC_AMT_PAYMENT_MEAN      = ("AMT_PAYMENT_CURRENT",   "mean"),
        CC_SK_DPD_MEAN           = ("SK_DPD",                "mean"),
        CC_SK_DPD_MAX            = ("SK_DPD",                "max"),
    ).reset_index()
    df = df.merge(cc_agg, on="SK_ID_CURR", how="left")
    print(f"After credit_card merge: {df.shape}")

    # POS CASH balance
    pos = pd.read_csv(f"{base_path}\\POS_CASH_balance.csv")
    pos_agg = pos.groupby("SK_ID_CURR").agg(
        POS_COUNT                = ("SK_ID_PREV",        "count"),
        POS_MONTHS_BALANCE_MEAN  = ("MONTHS_BALANCE",    "mean"),
        POS_CNT_INSTALMENT_MEAN  = ("CNT_INSTALMENT",    "mean"),
        POS_SK_DPD_MEAN          = ("SK_DPD",            "mean"),
        POS_SK_DPD_MAX           = ("SK_DPD",            "max"),
        POS_SK_DPD_DEF_MEAN      = ("SK_DPD_DEF",        "mean"),
    ).reset_index()
    df = df.merge(pos_agg, on="SK_ID_CURR", how="left")
    print(f"After POS_CASH merge: {df.shape}")

    print("All tables merged successfully!")
    return df

if __name__ == "__main__":
    df = load_and_merge()

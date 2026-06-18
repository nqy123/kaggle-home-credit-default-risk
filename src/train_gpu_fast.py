import gc
import json
import re
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "home-credit-default-risk"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ID_COL = "SK_ID_CURR"
TARGET = "TARGET"
N_SPLITS = 5
SEED = 42


def clean_names(df):
    used = {}
    cols = []
    for col in df.columns:
        name = re.sub(r"[^A-Za-z0-9_]+", "_", str(col))[:160]
        if name in used:
            used[name] += 1
            name = f"{name}_{used[name]}"
        else:
            used[name] = 0
        cols.append(name)
    df.columns = cols
    return df


def reduce_mem(df):
    for col in df.columns:
        if df[col].dtype == "object":
            continue
        if str(df[col].dtype).startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        else:
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def one_hot_small(df):
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return pd.get_dummies(df, columns=cat_cols, dummy_na=True)


def basic_agg(df, group_key, prefix, num_cols=None):
    if num_cols is None:
        num_cols = [c for c in df.columns if c != group_key and df[c].dtype != "object"]
    aggs = {c: ["mean", "max", "min", "sum"] for c in num_cols}
    out = df.groupby(group_key).agg(aggs)
    out.columns = [f"{prefix}_{c}_{s}".upper() for c, s in out.columns]
    out = out.reset_index()
    return reduce_mem(out)


def application():
    train = pd.read_csv(DATA_DIR / "application_train.csv")
    test = pd.read_csv(DATA_DIR / "application_test.csv")
    y = train[TARGET].copy()
    test_ids = test[ID_COL].copy()
    df = pd.concat([train, test], axis=0, ignore_index=True)

    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
    df["APP_MISSING_COUNT"] = df.isna().sum(axis=1)
    df["APP_CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["APP_ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["APP_GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1)
    df["APP_ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    df["APP_INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["APP_CREDIT_PER_PERSON"] = df["AMT_CREDIT"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["APP_EMPLOYED_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1)
    df["APP_EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["APP_EXT_SOURCE_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    df["APP_EXT_SOURCE_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["APP_DOC_SUM"] = df[[c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]].sum(axis=1)

    df = one_hot_small(df)
    y_out = df[TARGET].iloc[: len(train)].copy()
    df = df.drop(columns=[TARGET])
    df = reduce_mem(df)
    return df.iloc[: len(train)].reset_index(drop=True), df.iloc[len(train):].reset_index(drop=True), y_out, test_ids


def bureau_light():
    bureau = reduce_mem(pd.read_csv(DATA_DIR / "bureau.csv"))
    bureau["BURO_DEBT_CREDIT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["BURO_OVERDUE_CREDIT_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["BURO_DURATION"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_CREDIT"]
    status_counts = pd.crosstab(bureau[ID_COL], bureau["CREDIT_ACTIVE"]).reset_index()
    status_counts.columns = [ID_COL] + [f"BURO_ACTIVE_COUNT_{c}" for c in status_counts.columns[1:]]
    type_counts = pd.crosstab(bureau[ID_COL], bureau["CREDIT_TYPE"]).reset_index()
    type_counts.columns = [ID_COL] + [f"BURO_TYPE_COUNT_{c}" for c in type_counts.columns[1:]]
    num_cols = [
        "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT",
        "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE",
        "DAYS_CREDIT_UPDATE", "AMT_ANNUITY", "BURO_DEBT_CREDIT_RATIO",
        "BURO_OVERDUE_CREDIT_RATIO", "BURO_DURATION",
    ]
    out = basic_agg(bureau[[ID_COL] + num_cols], ID_COL, "BURO", num_cols)
    out = out.merge(status_counts, on=ID_COL, how="left").merge(type_counts, on=ID_COL, how="left")
    del bureau
    gc.collect()
    return reduce_mem(out)


def previous_light():
    prev = reduce_mem(pd.read_csv(DATA_DIR / "previous_application.csv"))
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        prev[c].replace(365243, np.nan, inplace=True)
    prev["PREV_CREDIT_APPLICATION_RATIO"] = prev["AMT_CREDIT"] / (prev["AMT_APPLICATION"] + 1)
    prev["PREV_ANNUITY_CREDIT_RATIO"] = prev["AMT_ANNUITY"] / (prev["AMT_CREDIT"] + 1)
    prev["PREV_DOWN_PAYMENT_RATIO"] = prev["AMT_DOWN_PAYMENT"] / (prev["AMT_CREDIT"] + 1)
    status_counts = pd.crosstab(prev[ID_COL], prev["NAME_CONTRACT_STATUS"]).reset_index()
    status_counts.columns = [ID_COL] + [f"PREV_STATUS_COUNT_{c}" for c in status_counts.columns[1:]]
    num_cols = [
        "AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE", "HOUR_APPR_PROCESS_START", "NFLAG_LAST_APPL_IN_DAY",
        "RATE_DOWN_PAYMENT", "DAYS_DECISION", "CNT_PAYMENT",
        "PREV_CREDIT_APPLICATION_RATIO", "PREV_ANNUITY_CREDIT_RATIO", "PREV_DOWN_PAYMENT_RATIO",
    ]
    out = basic_agg(prev[[ID_COL] + num_cols], ID_COL, "PREV", num_cols)
    out = out.merge(status_counts, on=ID_COL, how="left")
    del prev
    gc.collect()
    return reduce_mem(out)


def pos_light():
    pos = reduce_mem(pd.read_csv(DATA_DIR / "POS_CASH_balance.csv"))
    pos["POS_REMAINING_RATIO"] = pos["CNT_INSTALMENT_FUTURE"] / (pos["CNT_INSTALMENT"] + 1)
    status_counts = pd.crosstab(pos[ID_COL], pos["NAME_CONTRACT_STATUS"]).reset_index()
    status_counts.columns = [ID_COL] + [f"POS_STATUS_COUNT_{c}" for c in status_counts.columns[1:]]
    num_cols = ["MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF", "POS_REMAINING_RATIO"]
    out = basic_agg(pos[[ID_COL] + num_cols], ID_COL, "POS", num_cols)
    out = out.merge(status_counts, on=ID_COL, how="left")
    del pos
    gc.collect()
    return reduce_mem(out)


def installments_light():
    ins = reduce_mem(pd.read_csv(DATA_DIR / "installments_payments.csv"))
    ins["INS_PAYMENT_RATIO"] = ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 1)
    ins["INS_PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["INS_DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["INS_DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    num_cols = [
        "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
        "INS_PAYMENT_RATIO", "INS_PAYMENT_DIFF", "INS_DPD", "INS_DBD",
    ]
    out = basic_agg(ins[[ID_COL] + num_cols], ID_COL, "INS", num_cols)
    del ins
    gc.collect()
    return reduce_mem(out)


def credit_card_light():
    cc = reduce_mem(pd.read_csv(DATA_DIR / "credit_card_balance.csv"))
    cc["CC_BALANCE_LIMIT_RATIO"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["CC_DRAWING_LIMIT_RATIO"] = cc["AMT_DRAWINGS_CURRENT"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["CC_PAYMENT_MIN_RATIO"] = cc["AMT_PAYMENT_CURRENT"] / (cc["AMT_INST_MIN_REGULARITY"] + 1)
    status_counts = pd.crosstab(cc[ID_COL], cc["NAME_CONTRACT_STATUS"]).reset_index()
    status_counts.columns = [ID_COL] + [f"CC_STATUS_COUNT_{c}" for c in status_counts.columns[1:]]
    num_cols = [c for c in cc.columns if c not in [ID_COL, "SK_ID_PREV", "NAME_CONTRACT_STATUS"]]
    out = basic_agg(cc[[ID_COL] + num_cols], ID_COL, "CC", num_cols)
    out = out.merge(status_counts, on=ID_COL, how="left")
    del cc
    gc.collect()
    return reduce_mem(out)


def build_dataset():
    train_x, test_x, y, test_ids = application()
    all_x = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    all_x[ID_COL] = all_x[ID_COL].astype("int64")
    for name, fn in [
        ("bureau", bureau_light),
        ("previous", previous_light),
        ("pos", pos_light),
        ("installments", installments_light),
        ("credit_card", credit_card_light),
    ]:
        print(f"building {name}...")
        feat = fn()
        feat[ID_COL] = feat[ID_COL].astype("int64")
        all_x = all_x.merge(feat, on=ID_COL, how="left")
        print(f"after {name}: {all_x.shape}")
        del feat
        gc.collect()
    all_x = clean_names(all_x)
    all_x = reduce_mem(all_x)
    train_out = all_x.iloc[: len(train_x)].reset_index(drop=True)
    test_out = all_x.iloc[len(train_x):].reset_index(drop=True)
    return train_out, test_out, y.reset_index(drop=True), test_ids


def train_lgbm(train_x, test_x, y, test_ids):
    for df in [train_x, test_x]:
        for col in df.columns:
            if df[col].dtype == "bool":
                df[col] = df[col].astype(np.int8)
            elif df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    features = [c for c in train_x.columns if c != ID_COL]

    base_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 5000,
        "learning_rate": 0.02,
        "num_leaves": 48,
        "max_depth": 8,
        "min_child_samples": 220,
        "subsample": 0.82,
        "subsample_freq": 1,
        "colsample_bytree": 0.72,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    gpu_params = dict(base_params)
    gpu_params.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0, "max_bin": 255})

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train_x))
    pred = np.zeros(len(test_x))
    scores = []
    used_device = []

    for fold, (tr_idx, va_idx) in enumerate(folds.split(train_x, y), 1):
        print(f"\n========== Fold {fold} ==========")
        params = gpu_params
        model = lgb.LGBMClassifier(**params)
        try:
            model.fit(
                train_x.iloc[tr_idx][features],
                y.iloc[tr_idx],
                eval_set=[(train_x.iloc[va_idx][features], y.iloc[va_idx])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(250), lgb.log_evaluation(200)],
            )
            device = "gpu"
        except Exception as exc:
            print(f"[WARN] LightGBM GPU failed, fallback CPU: {exc}")
            model = lgb.LGBMClassifier(**base_params)
            model.fit(
                train_x.iloc[tr_idx][features],
                y.iloc[tr_idx],
                eval_set=[(train_x.iloc[va_idx][features], y.iloc[va_idx])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(250), lgb.log_evaluation(200)],
            )
            device = "cpu"
        va_pred = model.predict_proba(train_x.iloc[va_idx][features], num_iteration=model.best_iteration_)[:, 1]
        te_pred = model.predict_proba(test_x[features], num_iteration=model.best_iteration_)[:, 1]
        auc = roc_auc_score(y.iloc[va_idx], va_pred)
        print(f"fold {fold}: AUC={auc:.6f}, best_iter={model.best_iteration_}, device={device}")
        oof[va_idx] = va_pred
        pred += te_pred / N_SPLITS
        scores.append({"fold": fold, "auc": float(auc), "best_iteration": int(model.best_iteration_), "device": device})
        used_device.append(device)

    oof_auc = roc_auc_score(y, oof)
    print(f"\nOOF AUC={oof_auc:.6f}")
    pd.DataFrame({ID_COL: train_x[ID_COL], TARGET: oof}).to_csv(OUTPUT_DIR / "oof_lgbm_gpu_fast.csv", index=False)
    pd.DataFrame({ID_COL: test_ids, TARGET: pred}).to_csv(OUTPUT_DIR / "pred_lgbm_gpu_fast.csv", index=False)
    pd.DataFrame({ID_COL: test_ids, TARGET: pred}).to_csv(OUTPUT_DIR / "submission_lgbm_gpu_fast.csv", index=False)
    summary = {
        "model": "LightGBM GPU Fast Relational FE",
        "oof_auc": float(oof_auc),
        "fold_scores": scores,
        "devices": used_device,
        "n_features": len(features),
        "submission_file": "outputs/submission_lgbm_gpu_fast.csv",
        "notes": "轻量可解释关系表聚合：主表负担比率、bureau 历史信用、previous 申请、POS、installments、credit card 行为特征",
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return oof_auc


def main():
    train_x, test_x, y, test_ids = build_dataset()
    print(f"final train={train_x.shape}, test={test_x.shape}")
    train_lgbm(train_x, test_x, y, test_ids)


if __name__ == "__main__":
    main()

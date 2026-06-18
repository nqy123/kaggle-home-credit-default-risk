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


def reduce_mem(df):
    """降低内存占用，Home Credit 多表很大，这一步很关键。"""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "object":
            continue
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type).startswith("int"):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.float32)
    return df


def one_hot(df):
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    return df


def agg_numeric(df, group_key, prefix, aggs):
    grouped = df.groupby(group_key).agg(aggs)
    grouped.columns = [f"{prefix}_{col}_{stat}".upper() for col, stat in grouped.columns]
    grouped = grouped.reset_index()
    return grouped


def application_features():
    train = pd.read_csv(DATA_DIR / "application_train.csv")
    test = pd.read_csv(DATA_DIR / "application_test.csv")
    y = train[TARGET].copy()
    train_ids = train[ID_COL].copy()
    test_ids = test[ID_COL].copy()
    df = pd.concat([train, test], axis=0, ignore_index=True)

    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
    df["CODE_GENDER"].replace("XNA", np.nan, inplace=True)
    df["APP_MISSING_COUNT"] = df.isna().sum(axis=1)

    # 主表核心可解释风险比率。
    df["APP_CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["APP_ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["APP_GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1)
    df["APP_ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    df["APP_INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (df["CNT_CHILDREN"] + 1)
    df["APP_INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["APP_CREDIT_PER_PERSON"] = df["AMT_CREDIT"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["APP_EMPLOYED_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1)
    df["APP_REGISTRATION_BIRTH_RATIO"] = df["DAYS_REGISTRATION"] / (df["DAYS_BIRTH"] + 1)
    df["APP_ID_BIRTH_RATIO"] = df["DAYS_ID_PUBLISH"] / (df["DAYS_BIRTH"] + 1)
    df["APP_EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["APP_EXT_SOURCE_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    df["APP_EXT_SOURCE_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["APP_DOC_SUM"] = df[[c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]].sum(axis=1)

    df = one_hot(df)
    y_out = df[TARGET].iloc[: len(train)].copy()
    df = df.drop(columns=[TARGET])
    df = reduce_mem(df)
    return df.iloc[: len(train)].reset_index(drop=True), df.iloc[len(train) :].reset_index(drop=True), y_out, train_ids, test_ids


def bureau_features():
    bureau = reduce_mem(pd.read_csv(DATA_DIR / "bureau.csv"))
    bb = reduce_mem(pd.read_csv(DATA_DIR / "bureau_balance.csv"))

    bb = one_hot(bb)
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(["min", "max", "mean", "size"])
    bb_agg.columns = [f"BB_{a}_{b}".upper() for a, b in bb_agg.columns]
    bb_agg = bb_agg.reset_index()
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    del bb, bb_agg
    gc.collect()

    bureau["BURO_CREDIT_DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["BURO_CREDIT_OVERDUE_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["BURO_CREDIT_DURATION"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_CREDIT"]
    bureau["BURO_ENDDATE_FACT_DIFF"] = bureau["DAYS_ENDDATE_FACT"] - bureau["DAYS_CREDIT_ENDDATE"]

    bureau = one_hot(bureau)
    num_cols = [c for c in bureau.columns if c not in ["SK_ID_CURR", "SK_ID_BUREAU"]]
    aggs = {c: ["mean", "max", "min", "sum", "var"] for c in num_cols}
    buro_agg = agg_numeric(bureau.drop(columns=["SK_ID_BUREAU"]), ID_COL, "BURO", aggs)

    active = bureau[bureau.get("CREDIT_ACTIVE_Active", 0) == 1]
    closed = bureau[bureau.get("CREDIT_ACTIVE_Closed", 0) == 1]
    if len(active):
        active_agg = agg_numeric(active.drop(columns=["SK_ID_BUREAU"]), ID_COL, "ACTIVE", aggs)
        buro_agg = buro_agg.merge(active_agg, on=ID_COL, how="left")
    if len(closed):
        closed_agg = agg_numeric(closed.drop(columns=["SK_ID_BUREAU"]), ID_COL, "CLOSED", aggs)
        buro_agg = buro_agg.merge(closed_agg, on=ID_COL, how="left")
    del bureau, active, closed
    gc.collect()
    return reduce_mem(buro_agg)


def previous_features():
    prev = reduce_mem(pd.read_csv(DATA_DIR / "previous_application.csv"))
    prev["DAYS_FIRST_DRAWING"].replace(365243, np.nan, inplace=True)
    prev["DAYS_FIRST_DUE"].replace(365243, np.nan, inplace=True)
    prev["DAYS_LAST_DUE_1ST_VERSION"].replace(365243, np.nan, inplace=True)
    prev["DAYS_LAST_DUE"].replace(365243, np.nan, inplace=True)
    prev["DAYS_TERMINATION"].replace(365243, np.nan, inplace=True)

    prev["PREV_CREDIT_APPLICATION_RATIO"] = prev["AMT_CREDIT"] / (prev["AMT_APPLICATION"] + 1)
    prev["PREV_ANNUITY_CREDIT_RATIO"] = prev["AMT_ANNUITY"] / (prev["AMT_CREDIT"] + 1)
    prev["PREV_DOWN_PAYMENT_RATIO"] = prev["AMT_DOWN_PAYMENT"] / (prev["AMT_CREDIT"] + 1)
    prev["PREV_GOODS_CREDIT_RATIO"] = prev["AMT_GOODS_PRICE"] / (prev["AMT_CREDIT"] + 1)

    prev = one_hot(prev)
    num_cols = [c for c in prev.columns if c not in ["SK_ID_CURR", "SK_ID_PREV"]]
    aggs = {c: ["mean", "max", "min", "sum", "var"] for c in num_cols}
    prev_agg = agg_numeric(prev.drop(columns=["SK_ID_PREV"]), ID_COL, "PREV", aggs)

    for status in ["Approved", "Refused"]:
        col = f"NAME_CONTRACT_STATUS_{status}"
        if col in prev.columns:
            part = prev[prev[col] == 1]
            part_agg = agg_numeric(part.drop(columns=["SK_ID_PREV"]), ID_COL, f"PREV_{status.upper()}", aggs)
            prev_agg = prev_agg.merge(part_agg, on=ID_COL, how="left")
    del prev
    gc.collect()
    return reduce_mem(prev_agg)


def pos_features():
    pos = reduce_mem(pd.read_csv(DATA_DIR / "POS_CASH_balance.csv"))
    pos["POS_REMAINING_RATIO"] = pos["CNT_INSTALMENT_FUTURE"] / (pos["CNT_INSTALMENT"] + 1)
    pos = one_hot(pos)
    num_cols = [c for c in pos.columns if c not in ["SK_ID_CURR", "SK_ID_PREV"]]
    aggs = {c: ["mean", "max", "min", "sum", "var"] for c in num_cols}
    out = agg_numeric(pos.drop(columns=["SK_ID_PREV"]), ID_COL, "POS", aggs)
    del pos
    gc.collect()
    return reduce_mem(out)


def installments_features():
    ins = reduce_mem(pd.read_csv(DATA_DIR / "installments_payments.csv"))
    ins["INS_PAYMENT_RATIO"] = ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 1)
    ins["INS_PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["INS_DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["INS_DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    num_cols = [c for c in ins.columns if c not in ["SK_ID_CURR", "SK_ID_PREV"]]
    aggs = {c: ["mean", "max", "min", "sum", "var"] for c in num_cols}
    out = agg_numeric(ins.drop(columns=["SK_ID_PREV"]), ID_COL, "INS", aggs)
    del ins
    gc.collect()
    return reduce_mem(out)


def credit_card_features():
    cc = reduce_mem(pd.read_csv(DATA_DIR / "credit_card_balance.csv"))
    cc["CC_BALANCE_LIMIT_RATIO"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["CC_DRAWING_LIMIT_RATIO"] = cc["AMT_DRAWINGS_CURRENT"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["CC_PAYMENT_MIN_RATIO"] = cc["AMT_PAYMENT_CURRENT"] / (cc["AMT_INST_MIN_REGULARITY"] + 1)
    cc = one_hot(cc)
    num_cols = [c for c in cc.columns if c not in ["SK_ID_CURR", "SK_ID_PREV"]]
    aggs = {c: ["mean", "max", "min", "sum", "var"] for c in num_cols}
    out = agg_numeric(cc.drop(columns=["SK_ID_PREV"]), ID_COL, "CC", aggs)
    del cc
    gc.collect()
    return reduce_mem(out)


def build_dataset():
    train_x, test_x, y, train_ids, test_ids = application_features()
    all_x = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    for name, fn in [
        ("bureau", bureau_features),
        ("previous", previous_features),
        ("pos", pos_features),
        ("installments", installments_features),
        ("credit_card", credit_card_features),
    ]:
        print(f"building {name} features...")
        feat = fn()
        all_x = all_x.merge(feat, on=ID_COL, how="left")
        print(f"after {name}: {all_x.shape}")
        del feat
        gc.collect()

    all_x = reduce_mem(all_x)
    train_x = all_x.iloc[: len(train_x)].reset_index(drop=True)
    test_x = all_x.iloc[len(train_x) :].reset_index(drop=True)
    return train_x, test_x, y.reset_index(drop=True), train_ids, test_ids


def train_model(train_x, test_x, y, test_ids):
    # 多表聚合后，某些全空 dummy 聚合列会被 pandas 还原成 object；
    # 训练前统一转成数值，避免 LightGBM dtype 报错。
    for df in [train_x, test_x]:
        for col in df.columns:
            if df[col].dtype == "bool":
                df[col] = df[col].astype(np.int8)
            elif df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    clean_cols = []
    used = {}
    for col in train_x.columns:
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(col))
        clean = clean[:180]
        if clean in used:
            used[clean] += 1
            clean = f"{clean}_{used[clean]}"
        else:
            used[clean] = 0
        clean_cols.append(clean)
    train_x.columns = clean_cols
    test_x.columns = clean_cols
    features = [c for c in train_x.columns if c != ID_COL]
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 6000,
        "learning_rate": 0.015,
        "num_leaves": 34,
        "max_depth": 8,
        "min_child_samples": 180,
        "subsample": 0.82,
        "subsample_freq": 1,
        "colsample_bytree": 0.72,
        "reg_alpha": 0.2,
        "reg_lambda": 0.8,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train_x))
    pred = np.zeros(len(test_x))
    scores = []
    importances = []

    for fold, (tr_idx, va_idx) in enumerate(folds.split(train_x, y), 1):
        print(f"\n========== Fold {fold} ==========")
        model = lgb.LGBMClassifier(**params)
        model.fit(
            train_x.iloc[tr_idx][features],
            y.iloc[tr_idx],
            eval_set=[(train_x.iloc[va_idx][features], y.iloc[va_idx])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(300), lgb.log_evaluation(200)],
        )
        va_pred = model.predict_proba(train_x.iloc[va_idx][features], num_iteration=model.best_iteration_)[:, 1]
        te_pred = model.predict_proba(test_x[features], num_iteration=model.best_iteration_)[:, 1]
        auc = roc_auc_score(y.iloc[va_idx], va_pred)
        print(f"fold {fold}: AUC={auc:.6f}, best_iter={model.best_iteration_}")
        oof[va_idx] = va_pred
        pred += te_pred / N_SPLITS
        scores.append({"fold": fold, "auc": float(auc), "best_iteration": int(model.best_iteration_)})
        importances.append(pd.DataFrame({"feature": features, "importance": model.feature_importances_, "fold": fold}))

    oof_auc = roc_auc_score(y, oof)
    print(f"\nOOF AUC={oof_auc:.6f}")
    pd.DataFrame({ID_COL: train_x[ID_COL], TARGET: oof}).to_csv(OUTPUT_DIR / "oof_lgbm_best.csv", index=False)
    pd.DataFrame({ID_COL: test_ids, TARGET: pred}).to_csv(OUTPUT_DIR / "pred_lgbm_best.csv", index=False)
    pd.DataFrame({ID_COL: test_ids, TARGET: pred}).to_csv(OUTPUT_DIR / "submission_lgbm_best.csv", index=False)
    pd.concat(importances).groupby("feature")["importance"].mean().sort_values(ascending=False).head(200).to_csv(
        OUTPUT_DIR / "feature_importance_top200.csv"
    )
    summary = {
        "model": "LightGBM full relational FE",
        "metric": "AUC",
        "oof_auc": float(oof_auc),
        "fold_scores": scores,
        "n_features": len(features),
        "submission_file": "outputs/submission_lgbm_best.csv",
        "notes": "主表可解释比率 + bureau/previous/POS/installments/credit_card 聚合特征",
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return oof_auc


def main():
    train_x, test_x, y, train_ids, test_ids = build_dataset()
    print(f"final train={train_x.shape}, test={test_x.shape}")
    train_model(train_x, test_x, y, test_ids)


if __name__ == "__main__":
    main()

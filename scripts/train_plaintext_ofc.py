# train_plaintext_ofc.py
# OPTIMIZED FOR CLOUD (OFC)
# 1. Removed threading backend (fixes GIL bottleneck).
# 2. Swapped parallelism: Models use all cores (n_jobs=-1), Search runs sequentially (n_jobs=1).

import os, json, argparse, warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ----------------- Model configs (grids) -----------------
TARGET_PRECISION = 0.95
SEED = 42

MODEL_CFGS = {
    "rf": {
        "coarse": {
            "n_iter": 30, "cv_folds": 2,
            "param_grid": {
                "n_estimators": [300, 600],
                "max_depth": [8, 12],
                "min_samples_leaf": [1, 3],
                "max_features": ["sqrt", 0.5],
                "class_weight": ["balanced", {0: 1, 1: 10}],
                "max_samples": [0.8, 1.0],
            },
        },
        "fine": {
            "n_iter": 60, "cv_folds": 5,
            "param_grid": {
                "n_estimators": [250, 300, 350, 400],
                "max_depth": [10, 12, 14],
                "min_samples_leaf": [1, 2],
                "min_samples_split": [2, 4, 6],
                "max_features": ["sqrt"],
                "class_weight": [{0: 1, 1: 8}, {0: 1, 1: 10}, {0: 1, 1: 12}],
                "max_samples": [0.9, 1.0],
            },
        },
        "final": {
            "n_iter": 1, "cv_folds": 10,
            "param_grid": {
                "n_estimators": [300],
                "max_depth": [12],
                "min_samples_leaf": [1],
                "min_samples_split": [2],
                "max_features": ["sqrt"],
                "class_weight": [{0: 1, 1: 10}],
                "max_samples": [1.0],
            },
        },
    },
    "xgb": {
        "coarse": {
            "n_iter": 30, "cv_folds": 2,
            "param_grid": {
                "n_estimators": [200, 400],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        },
        "fine": {
            "n_iter": 60, "cv_folds": 5,
            "param_grid": {
                "n_estimators": [300, 400, 500],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.9, 1.0],
                "colsample_bytree": [0.9, 1.0],
            },
        },
        "final": {
            "n_iter": 1, "cv_folds": 5,
            "param_grid": {
                "n_estimators": [400],
                "max_depth": [6],
                "learning_rate": [0.05],
                "subsample": [1.0],
                "colsample_bytree": [1.0],
            },
        },
    },
    "lr": {
        "coarse": {
            "n_iter": 10, "cv_folds": 3,
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
                "class_weight": ["balanced", None],
                "max_iter": [500, 1000],
            },
        },
        "fine": {
            "n_iter": 20, "cv_folds": 5,
            "param_grid": {
                "C": [0.05, 0.1, 0.5, 1.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
                "class_weight": ["balanced", None],
                "max_iter": [1000],
            },
        },
        "final": {
            "n_iter": 1, "cv_folds": 5,
            "param_grid": {
                "C": [0.1],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
                "class_weight": ["balanced"],
                "max_iter": [1000],
            },
        },
    },
}

# OPTIMIZATION: Internal parallelism for estimators
ESTIMATORS = {
    "rf": RandomForestClassifier(random_state=SEED, n_jobs=-1),
    "xgb": XGBClassifier(random_state=SEED, n_jobs=-1, tree_method="hist", eval_metric="logloss"),
    "lr": LogisticRegression(random_state=SEED, n_jobs=-1),
}

def threshold_for_precision(y_true, y_prob, p=0.95):
    P, R, T = precision_recall_curve(y_true, y_prob)
    idx = np.where(P[:-1] >= p)[0]
    if idx.size == 0:
        return 0.5, 0.0, 0.0
    best = idx[np.argmax(R[idx])]
    return float(T[best]), float(P[best]), float(R[best])

def train_one(model_key: str, mode: str):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = {"rf": "RandomForest", "xgb": "XGB", "lr": "LR"}[model_key]
    cfg = MODEL_CFGS[model_key][mode]
    estimator = ESTIMATORS[model_key]

    model_root = f"artifacts/{label}-PLTT"
    out_dir = f"{model_root}/{mode}/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    data_path = "train.parquet"
    if not os.path.exists(data_path):
        data_path = "data/train.parquet" # Local fallback 1
    if not os.path.exists(data_path):
        data_path = "/workspace/train.parquet" # Cloud fallback

    print(f"Loading data from: {data_path}")
    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find train.parquet")
        return

    # Target & features
    y = df["isFraud"]
    X = df.drop(columns=["isFraud", "TransactionDT"])

    for c in X.columns:
        if np.issubdtype(X[c].dtype, np.floating):
            X[c] = X[c].astype(np.float32)
        elif np.issubdtype(X[c].dtype, np.integer):
            X[c] = X[c].astype(np.int32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    ct = ColumnTransformer([("scale_amount", StandardScaler(), ["TransactionAmt"])], remainder="passthrough")
    pipe = Pipeline(steps=[("pre", ct), ("clf", estimator)])

    param_space = {f"clf__{k}": v for k, v in cfg["param_grid"].items()}
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=SEED)
    
    # OPTIMIZATION: n_jobs=1 for search, since model has n_jobs=-1
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_space,
        n_iter=cfg["n_iter"],
        cv=cv,
        random_state=SEED,
        scoring="average_precision",
        n_jobs=1,
        refit=True,
        verbose=2,
    )

    print("Starting Grid Search...")
    search.fit(X_tr, y_tr)
    best = search.best_estimator_

    y_prob = best.predict_proba(X_te)[:, 1]
    thr, P95, R95 = threshold_for_precision(y_te, y_prob, TARGET_PRECISION)
    prauc = average_precision_score(y_te, y_prob)
    roc_auc = roc_auc_score(y_te, y_prob)

    summary = {
        "run": {"mode": mode, "run_id": run_id},
        "model": label,
        "PR-AUC": float(prauc),
        "ROC-AUC": float(roc_auc),
        "thr_at_P>=%.2f" % TARGET_PRECISION: float(thr),
        "precision_at_thr": float(P95),
        "recall_at_thr": float(R95),
        "best_params": search.best_params_,
    }

    pd.DataFrame(search.cv_results_).to_csv(os.path.join(out_dir, f"{label}_cv_results.csv"), index=False)
    with open(os.path.join(out_dir, f"{label}_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    joblib.dump(best, os.path.join(out_dir, f"{label}_best.joblib"))

    runlog_path = os.path.join(model_root, "run_log.txt")
    with open(runlog_path, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] model={model_key} mode={mode} run_id={run_id} PRAUC={prauc:.4f}\n")

    print("\nSaved artifacts to:", out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["rf", "xgb", "lr"])
    parser.add_argument("--mode", required=True, choices=["coarse", "fine", "final"])
    args = parser.parse_args()
    train_one(args.model.lower(), args.mode.lower())

if __name__ == "__main__":
    main()

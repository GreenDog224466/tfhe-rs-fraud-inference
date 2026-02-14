import argparse
import pandas as pd
import numpy as np
import time
import os
import json
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

def load_data(path):
    print(f"[INFO] Loading data from {path}...")
    df = pd.read_parquet(path)
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    print(f"[INFO] Data Shape: {X.shape}")
    return X, y

def main():
    parser = argparse.ArgumentParser()
    # ✅ FIX: Changed from absolute cloud mounts (/mnt/pv/...) to relative paths
    parser.add_argument("--data_path", type=str, default="data/processed/train.parquet")
    parser.add_argument("--output_base", type=str, default="artifacts/RandomForest-PLTT/fine")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_base, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    X, y = load_data(args.data_path)
    
    # INNER: 1 Thread (Keeps memory low per model)
    clf = RandomForestClassifier(n_jobs=1, class_weight={0: 1, 1: 10}, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[('scale_amount', StandardScaler(), ['TransactionAmt'])],
        remainder='passthrough'
    )
    
    pipeline = Pipeline([('pre', preprocessor), ('clf', clf)])

    # GRID: 162 Fits
    param_grid = {
        'clf__n_estimators' : [10],
        'clf__max_depth' : [2],
        'clf__max_features': [0.4, 0.5, 0.6],
        'clf__min_samples_leaf': [1, 2]
    }
    
    # OUTER: 20 Workers (High Speed, now Safe with 512GB RAM)
    search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        scoring='average_precision', 
        cv=2, 
        verbose=2, 
        n_jobs=10, 
        return_train_score=False
    )
    
    print(f"[INFO] Starting Grid Search (20 workers)...")
    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start
    
    # METRICS & SAVE
    print("[INFO] Calculating metrics...")
    best_model = search.best_estimator_
    y_pred = best_model.predict_proba(X)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y, y_pred)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y, y_pred)
    
    # Threshold @ P95
    target_precision = 0.95
    valid_indices = np.where(precision >= target_precision)[0]
    if len(valid_indices) > 0:
        idx = valid_indices[0]
        thr_at_p95 = float(thresholds[idx]) if idx < len(thresholds) else 1.0
        rec_at_p95 = float(recall[idx])
        prec_at_p95 = float(precision[idx])
    else:
        thr_at_p95, rec_at_p95, prec_at_p95 = 0.0, 0.0, 0.0

    metrics = {
        "run_id": timestamp,
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "thr_at_P>=0.95": thr_at_p95,
        "precision_at_thr": prec_at_p95,
        "recall_at_thr": rec_at_p95,
        "best_params": search.best_params_,
        "training_time": elapsed
    }

    with open(os.path.join(output_dir, "RF_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    joblib.dump(best_model, os.path.join(output_dir, "RF_best.joblib"), compress=3)
    pd.DataFrame(search.cv_results_).to_csv(os.path.join(output_dir, "RF_cv_results.csv"), index=False)
    print(f"[SUCCESS] Saved to {output_dir}")

if __name__ == "__main__":
    main()

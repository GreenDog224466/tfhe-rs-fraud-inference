#!/usr/bin/env python3
"""
Preprocessing for IEEE-CIS Fraud Detection dataset.

Outputs:
 - data/processed/train.parquet       (merged & preprocessed dataset)
 - artifacts/encoders/*.joblib        (imputers, scaler, encoder)
 - data/processed/metadata.json       (metadata about features)
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib

# -------------------------
# Paths
# -------------------------
raw_dir = Path("data/raw")
processed_dir = Path("data/processed")
artifacts_dir = Path("artifacts/encoders")

# Create folders if they don't exist
processed_dir.mkdir(parents=True, exist_ok=True)
artifacts_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load CSVs
# -------------------------
print("Loading train_transaction.csv...")
tx = pd.read_csv(raw_dir / "train_transaction.csv", low_memory=False)
print(f"✓ Loaded transaction data: {tx.shape[0]:,} rows, {tx.shape[1]} columns")

print("Loading train_identity.csv...")
idf = pd.read_csv(raw_dir / "train_identity.csv", low_memory=False)
print(f"✓ Loaded identity data: {idf.shape[0]:,} rows, {idf.shape[1]} columns")

# -------------------------
# Merge on TransactionID
# -------------------------
print("Merging transaction and identity data...")
df = tx.merge(idf, how="left", on="TransactionID")
print(f"✓ Merged dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

# -------------------------
# Separate label
# -------------------------
y = df["isFraud"].astype(int)

# -------------------------
# Identify numeric and categorical features
# -------------------------
print("Identifying feature types...")
exclude = {"TransactionID", "isFraud"}
all_cols = [c for c in df.columns if c not in exclude]

categorical_cols = [c for c in all_cols if df[c].dtype == "object"]
numeric_cols = [c for c in all_cols if c not in categorical_cols]
print(f"✓ Found {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features")

# -------------------------
# Handle missing values
# -------------------------
print("Imputing missing values...")
# Numeric: median
print("  - Imputing numeric features (median)...")
num_imputer = SimpleImputer(strategy="median")
X_num = num_imputer.fit_transform(df[numeric_cols])

# Categorical: fill "MISSING"
print("  - Imputing categorical features (MISSING)...")
cat_imputer = SimpleImputer(strategy="constant", fill_value="MISSING")
X_cat = cat_imputer.fit_transform(df[categorical_cols].astype(str))
print("✓ Missing values imputed")

# -------------------------
# Encode categorical features
# -------------------------
print("Encoding categorical features...")
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_cat_enc = ord_enc.fit_transform(X_cat)
print("✓ Categorical features encoded")

# -------------------------
# Scale numeric features
# -------------------------
print("Scaling numeric features...")
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)
print("✓ Numeric features scaled")

# -------------------------
# Combine all features
# -------------------------
print("Combining all features...")
processed = pd.concat([
    df[["TransactionID"]].reset_index(drop=True),
    pd.DataFrame(X_num_scaled, columns=numeric_cols),
    pd.DataFrame(X_cat_enc, columns=categorical_cols)
], axis=1)

processed["isFraud"] = y.values
print(f"✓ Combined dataset: {processed.shape[0]:,} rows, {processed.shape[1]} columns")

# -------------------------
# Save processed dataset
# -------------------------
print(f"Saving processed dataset to {processed_dir / 'train.parquet'}...")
processed.to_parquet(processed_dir / "train.parquet", index=False)
print(f"✓ Merged and processed dataset saved to {processed_dir / 'train.parquet'}")

# -------------------------
# Save preprocessing artifacts
# -------------------------
print("Saving preprocessing artifacts...")
joblib.dump(num_imputer, artifacts_dir / "num_imputer.joblib")
joblib.dump(cat_imputer, artifacts_dir / "cat_imputer.joblib")
joblib.dump(ord_enc, artifacts_dir / "ordinal_encoder.joblib")
joblib.dump(scaler, artifacts_dir / "scaler.joblib")
print("✓ Preprocessing artifacts saved")

# -------------------------
# Save metadata
# -------------------------
metadata = {
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "n_rows": int(processed.shape[0]),
    "n_columns": int(processed.shape[1]),
    "artifacts": {
        "num_imputer": str(artifacts_dir / "num_imputer.joblib"),
        "cat_imputer": str(artifacts_dir / "cat_imputer.joblib"),
        "ordinal_encoder": str(artifacts_dir / "ordinal_encoder.joblib"),
        "scaler": str(artifacts_dir / "scaler.joblib")
    }
}

print("Saving metadata...")
with open(processed_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved to {processed_dir / 'metadata.json'}")
print("\n✅ Preprocessing complete!")

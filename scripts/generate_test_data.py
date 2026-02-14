import polars as pl
import numpy as np
import os

# CONFIGURATION
# We need to match the Rust config exactly:
# - 433 Features (as per your JSON)
# - Integers (Int64)
# - Enough rows for the "Saturation" test (at least 20 rows)
NUM_ROWS = 100
NUM_FEATURES = 433
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = "scaled_features.parquet"

def generate_data():
    print(f"⚙️  Generating synthetic data: {NUM_ROWS} rows x {NUM_FEATURES} features...")
    
    # 1. Create Random Integers (simulating scaled features)
    # We use valid int64 range. 
    data = np.random.randint(-100, 100, size=(NUM_ROWS, NUM_FEATURES), dtype=np.int64)
    
    # 2. Create Column Names (feature_0, feature_1, ...)
    columns = [f"feature_{i}" for i in range(NUM_FEATURES)]
    
    # 3. Create Polars DataFrame
    df = pl.DataFrame(data, schema=columns)
    
    # 4. Ensure Directory Exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 5. Save to Parquet
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.write_parquet(output_path)
    
    print(f"✅ SUCCESS: Synthetic dataset saved to {output_path}")
    print("   -> You can now run 'cargo run --release'")

if __name__ == "__main__":
    generate_data()
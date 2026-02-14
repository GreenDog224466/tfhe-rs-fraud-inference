// src/he_client.rs
// Client-side homomorphic encryption workflow

use crate::config::*;
use crate::encryption::*;
use polars::prelude::*;
use std::path::Path;

/// Main entry point for client-side encryption
pub fn run_client_encryption() {
    println!("--- Starting client-side HE encryption ---");

    // 1. Generate or load the client key
    println!("Generating client key...");
    let client_key = generate_client_key();
    println!("Client key ready.");

    // 2. Load scaled features
    println!("Loading scaled features...");
    let df = load_features();
    println!("Loaded {} rows and {} columns.", df.height(), df.width());

    // 3. Optional: small-batch validation
    if ENABLE_LOGGING {
        println!("Validating encryption/decryption on a small batch...");
    }
    validate_encryption(&client_key);
    if ENABLE_LOGGING {
        println!("Validation successful.");
    }

    // 4. Encrypt full dataset in batches
    println!("Encrypting dataset in batches of {} rows...", BATCH_SIZE);
    let num_rows = df.height();
    let num_batches = (num_rows + BATCH_SIZE - 1) / BATCH_SIZE;

    for batch_idx in 0..num_batches {
        let start = batch_idx * BATCH_SIZE;
        let end = ((batch_idx + 1) * BATCH_SIZE).min(num_rows);
        let batch_df = df.slice(start as i64, (end - start) as usize);

        println!("Encrypting batch {} / {} (rows {}..{})", batch_idx + 1, num_batches, start, end);

        let encrypted_batch = encrypt_batch(&client_key, &batch_df);

        // Save batch to disk
        let output_path = Path::new(ENCRYPTED_OUTPUT_DIR)
            .join(format!("encrypted_batch_{}.bin", batch_idx + 1));
        save_encrypted_batch(&encrypted_batch, &output_path);

        if ENABLE_LOGGING {
            println!("Saved encrypted batch to {:?}", output_path);
        }
    }

    println!("--- All batches encrypted successfully ---");
}

/// Helper to load features from Parquet
fn load_features() -> DataFrame {
    let df = LazyFrame::scan_parquet(SCALED_FEATURES_PATH, Default::default())
        .expect("Failed to read Parquet")
        .collect()
        .expect("Failed to collect DataFrame");
    df
}

/// Helper to save encrypted batch to disk
fn save_encrypted_batch(batch: &Vec<FheInt32>, path: &std::path::Path) {
    let file = std::fs::File::create(path).expect("Failed to create output file");
    bincode::serialize_into(file, batch).expect("Failed to serialize encrypted batch");
}

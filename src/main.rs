// src/main.rs
mod config;
pub mod server;

use config::{AppConfig, FheType, HOURLY_COST_USD};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::error::Error;
use std::time::Instant;
use tfhe::prelude::*;
use tfhe::{generate_keys, set_server_key, ConfigBuilder};
use polars::prelude::*;
use rayon::prelude::*;

// --- Data Structures ---
#[derive(Debug, Deserialize)]
struct ModelWeights {
    bias: i64,
    weights: Vec<i64>,
}

#[derive(Debug)]
struct TestStage {
    features: usize,
    rows: usize,
    category: &'static str,
    description: &'static str,
}

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Initialize Configuration
    let config = AppConfig::new();
    
    println!("=========================================================================================");
    println!("🔐 FHE FRAUD DETECTION: ROBUSTNESS VALIDATION");
    println!("   Target: GCP N2-Standard-16 (AVX-512) or Apple Silicon");
    println!("   Data Path: {:?}", config.get_data_path());
    println!("=========================================================================================\n");

    // 2. Setup & Key Gen (Simulated Key Management)
    println!("[Init] Generating Cryptographic Parameters (tfhe-rs default)...");
    let fhe_config = ConfigBuilder::default().build();
    let (client_key, server_key) = generate_keys(fhe_config);
    set_server_key(server_key.clone());
    println!("   -> Keys Active.\n");

    // 3. Resource Loading (Safe Error Handling instead of .unwrap())
    let weight_path = config.get_weights_path();
    let file = File::open(&weight_path).map_err(|e| format!("Failed to open weights at {:?}: {}", weight_path, e))?;
    let reader = BufReader::new(file);
    let full_model: ModelWeights = serde_json::from_reader(reader)?;
    let max_features = full_model.weights.len();
    println!("[Init] Model Loaded. Max Features: {}", max_features);

    // Load Parquet (Lazy Execution)
    let data_path = config.get_data_path();
    let df = LazyFrame::scan_parquet(&data_path, ScanArgsParquet::default())?
        .collect()?;
    let total_rows_available = df.height();
    println!("[Init] Dataset Loaded. Available Rows: {}\n", total_rows_available);

    // 4. THE VALIDATION PYRAMID
    let stages = vec![
        TestStage { features: 5, rows: 5, category: "SANITY", description: "I/O Pipeline Check" },
        TestStage { features: max_features, rows: 1, category: "CRYPTO", description: "Noise Budget Check" },
        TestStage { features: max_features, rows: 16, category: "SATURATION", description: "1:1 Core Mapping" },
    ];

    println!("{:<12} | {:<28} | {:<10} | {:<10} | {:<12} | {:<10}", 
        "CATEGORY", "TEST SCENARIO", "TIME (s)", "ROWS/SEC", "COST ($)", "STATUS");
    println!("{}", "-".repeat(100));

    // 5. Execution Engine
    for stage in stages {
        if stage.features > max_features { 
            println!("Skipping stage (req {} features, have {})", stage.features, max_features);
            continue; 
        }

        let current_weights = &full_model.weights[0..stage.features];
        let current_bias = full_model.bias;
        let rows_to_process = std::cmp::min(stage.rows, total_rows_available);
        
        // Pre-fetch rows to avoid IO inside the timer
        let mut batch_plain_inputs: Vec<Vec<i64>> = Vec::with_capacity(rows_to_process);
        for r in 0..rows_to_process {
            let mut row_vec = Vec::with_capacity(stage.features);
            
            // GRACEFUL DEGRADATION: Handle potential row access failure safely
            if let Some(row_series) = df.get_row(r).ok() {
                for c in 0..stage.features {
                    if let AnyValue::Int64(val) = row_series.0[c] { 
                        row_vec.push(val); 
                    } else { 
                        row_vec.push(0); // Default to 0 on null/error instead of crashing
                    }
                }
                batch_plain_inputs.push(row_vec);
            }
        }

        let start_total = Instant::now();

        // A. Encryption (Parallel)
        let encrypted_batch: Vec<Vec<FheType>> = batch_plain_inputs.par_iter()
            .map(|row| {
                row.iter().map(|&val| FheType::encrypt(val, &client_key)).collect()
            })
            .collect();

        // B. Compute Dot Product (Parallel)
        let server_key_handle = server_key.clone();
        let encrypted_results: Vec<FheType> = encrypted_batch.par_iter()
            .map(|enc_row| {
                set_server_key(server_key_handle.clone());
                
                let mut accumulator = FheType::encrypt(current_bias, &client_key);
                for (w, enc_val) in current_weights.iter().zip(enc_row.iter()) {
                    accumulator = accumulator + (enc_val * *w);
                }
                accumulator
            })
            .collect();

        // C. Decrypt & Verify
        let mut success_count = 0;
        for (i, result_cipher) in encrypted_results.iter().enumerate() {
            let decrypted: i64 = result_cipher.decrypt(&client_key);
            let input_row = &batch_plain_inputs[i];
            let expected: i64 = input_row.iter().zip(current_weights.iter())
                .map(|(a, b)| a * b)
                .sum::<i64>() + current_bias;
            
            if decrypted == expected { success_count += 1; }
        }

        let duration = start_total.elapsed();
        let seconds = duration.as_secs_f64();
        let throughput = rows_to_process as f64 / seconds; 
        let cost_estimate = (HOURLY_COST_USD / 3600.0) * seconds;
        let status = if success_count == rows_to_process { "✅ PASS" } else { "❌ FAIL" };

        println!("{:<12} | {:<28} | {:<10.2} | {:<10.2} | ${:<11.6} | {}", 
            stage.category, stage.description, seconds, throughput, cost_estimate, status
        );
    }
    
    Ok(())
}
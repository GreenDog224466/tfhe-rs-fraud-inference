use tfhe::{ClientKey, ServerKey, FheUint64, ConfigBuilder, generate_keys as tfhe_gen_keys, set_server_key};
use tfhe::prelude::*; 
use polars::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

// --- 1. Key Generation ---
pub fn generate_keys() -> (ClientKey, ServerKey) {
    println!("Generating keys (this may take a moment)...");
    let config = ConfigBuilder::default().build();
    return tfhe_gen_keys(config);
}

// --- 2. Thread Pool Creator (The Fix) ---
// Instead of a global init, we create a specific pool for FHE work
pub fn create_fhe_pool(server_key: ServerKey) -> ThreadPool {
    let key_for_workers = server_key.clone();
    
    // Also set it for the current main thread just in case
    set_server_key(server_key);

    ThreadPoolBuilder::new()
        .start_handler(move |_| {
            // Every new thread created in this pool gets the key immediately
            set_server_key(key_for_workers.clone());
        })
        .build()
        .expect("Failed to create thread pool")
}

// --- 3. The Inference Operation ---
pub fn dot_product(row: &[FheUint64], weights: &[u64]) -> FheUint64 {
    // This will now automatically use the pool we created in main.rs
    row.par_iter()
        .zip(weights.par_iter())
        .map(|(enc_val, weight)| enc_val * *weight) 
        .reduce(
            || FheUint64::try_encrypt_trivial(0u64).unwrap(),
            |a, b| a + b
        )
}

// --- 4. Data Helper ---
pub fn generate_dummy_data(rows: usize, features: usize) -> DataFrame {
    let mut cols = Vec::new();
    let mut rng = rand::thread_rng();
    for i in 0..features {
        let name = format!("feat_{}", i);
        let data: Vec<u64> = (0..rows).map(|_| rng.gen_range(0..10)).collect();
        cols.push(Series::new(&name, data));
    }
    DataFrame::new(cols).unwrap()
}

// --- 5. Encryption Helper ---
pub fn encrypt_dataset_to_rows(client_key: &ClientKey, df: &DataFrame) -> Vec<Vec<FheUint64>> {
    let n_rows = df.height();
    let n_cols = df.width();
    let mut encrypted_rows = Vec::with_capacity(n_rows);

    for r in 0..n_rows {
        let mut row_vec = Vec::with_capacity(n_cols);
        let row_data = df.get_row(r).unwrap();
        for val in row_data.0 {
            let u64_val = match val {
                AnyValue::UInt64(v) => v,
                AnyValue::Int64(v) => v as u64,
                AnyValue::Int32(v) => v as u64,
                _ => 0,
            };
            row_vec.push(FheUint64::encrypt(u64_val, client_key));
        }
        encrypted_rows.push(row_vec);
    }
    encrypted_rows
}

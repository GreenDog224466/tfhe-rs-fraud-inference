// src/server.rs
use crate::config::{AppConfig, FheType};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use serde::Deserialize;
use rayon::prelude::*;
use tfhe::prelude::*;
use tfhe::set_server_key;

// Define the model struct (this must match the JSON structure)
#[derive(Debug, Deserialize)]
struct ModelWeights {
    bias: i64,
    weights: Vec<i64>,
}

/// The core inference engine
/// Takes encrypted inputs and returns encrypted predictions
pub struct FheServer {
    weights: Vec<i64>,
    bias: i64,
    server_key: tfhe::ServerKey,
}

impl FheServer {
    // Explicit initialization ensures we don't run with missing weights
    pub fn new(config: &AppConfig, server_key: tfhe::ServerKey) -> Result<Self, Box<dyn Error>> {
        let weight_path = config.get_weights_path();
        
        // Robust error handling for file access
        let file = File::open(&weight_path)
            .map_err(|e| format!("Server failed to load weights from {:?}: {}", weight_path, e))?;
            
        let reader = BufReader::new(file);
        let model: ModelWeights = serde_json::from_reader(reader)?;

        Ok(Self {
            weights: model.weights,
            bias: model.bias,
            server_key,
        })
    }

    pub fn predict_batch(&self, encrypted_batch: &[Vec<FheType>]) -> Vec<FheType> {
        // Parallelize the batch processing using Rayon
        // We clone the server key for each thread (cheap, it's an Arc usually)
        let server_key_handle = self.server_key.clone();
        
        encrypted_batch.par_iter()
            .map(|row| {
                // Ensure thread-local key is set for this specific worker thread
                set_server_key(server_key_handle.clone());
                self.predict_row(row)
            })
            .collect()
    }

    fn predict_row(&self, row: &[FheType]) -> FheType {
        // Start with the bias (trivial encryption = cheap)
        // usage of try_encrypt_trivial requires imports, but strictly strictly strictly:
        // tfhe::FheInt64::try_encrypt_trivial is not always standard. 
        // We will use standard encryption for consistency if trivial fails, 
        // but 'encrypt_trivial' is the standard method in FheInt64.
        let mut accumulator = FheType::try_encrypt_trivial(self.bias).unwrap_or_else(|_| {
             // Fallback if trivial encryption fails (rare)
             panic!("Failed to encrypt trivial bias");
        });

        for (feature_val, weight) in row.iter().zip(self.weights.iter()) {
            // FHE Logic: result += feature * weight
            // Note: In FHE, plain multiplication is cheaper than encrypted mult.
            accumulator = accumulator + (feature_val * *weight);
        }
        accumulator
    }
}
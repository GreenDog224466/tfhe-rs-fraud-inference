use std::path::PathBuf;
use std::env;

pub type FheType = tfhe::FheInt64; 
pub const HOURLY_COST_USD: f64 = 0.78;

pub struct AppConfig {
    pub base_path: PathBuf,
}

impl AppConfig {
    pub fn new() -> Self {
        let base_path = env::var("APP_BASE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::current_dir().unwrap_or(PathBuf::from(".")));
        AppConfig { base_path }
    }

    pub fn get_weights_path(&self) -> PathBuf {
        self.base_path.join("data/LR_weights_quantized.json")
    }

    pub fn get_data_path(&self) -> PathBuf {
        self.base_path.join("data/processed/scaled_features.parquet")
    }
}
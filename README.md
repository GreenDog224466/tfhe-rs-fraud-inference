# Privacy-Preserving Fraud Detection using FHE (IEEE-CIS)

This project implements a privacy-preserving machine learning inference system for fraud detection using **Fully Homomorphic Encryption (FHE)**. It allows a client to encrypt sensitive financial transaction data, send it to a server for analysis, and receive a fraud prediction without the server ever seeing the raw data.

The system is built using **Rust** for the high-performance FHE operations (using the `tfhe` library) and **Python** for model training and data preprocessing.

## 📂 Project Structure

* **`src/`**: The core Rust application code.
* `main.rs`: Entry point and **Validation Pyramid** test suite.
* `server.rs`: The core FHE inference engine logic.
* `config.rs`: Centralized configuration and path management.


* **`scripts/`**: Python scripts for training and data generation.
* `generate_test_data.py`: Creates synthetic data for local validation.
* `extract_lr_weights.py`: Extracts trained model weights into JSON format.
* `preprocess.py`: Handles feature scaling and cleaning.


* **`data/`**: Contains trained model weights and data files.
* `LR_weights_quantized.json`: The quantized weights used by the FHE server.



## 🧪 Validation & Benchmarking

The system features an automated **Validation Pyramid**. This allows a user to verify cryptographic integrity and hardware performance scaling in a single execution.

| Tier | Name | Features | Rows | Purpose |
| --- | --- | --- | --- | --- |
| **01** | **SANITY** | 5 | 5 | Verifies I/O, Parquet loading, and basic math accuracy. |
| **02** | **CRYPTO** | 433 | 1 | Tests noise budget with the full-scale feature set. |
| **03** | **SATURATION** | 433 | 16 | Benchmarks parallel throughput (16-core optimization). |

### 🏆 Latest Validation Results

The following results confirm that the system correctly handles full-scale FHE operations with zero data corruption. The **CRYPTO** pass specifically validates that the `tfhe-rs` noise budget is sufficient for the 433-feature model.

```text
CATEGORY     | TEST SCENARIO                | TIME (s)   | ROWS/SEC   | COST ($)     | STATUS    
----------------------------------------------------------------------------------------------------
SANITY       | I/O Pipeline Check           | 89.71      | 0.06       | $0.019438    | ✅ PASS
CRYPTO       | Noise Budget Check           | 2412.60    | 0.00       | $0.522730    | ✅ PASS

```

> **Note:** The `CRYPTO` stage confirms that the cryptographic parameters are mathematically sound for high-dimensional data (433 features).

## 🚀 Getting Started

### Prerequisites

* **Rust**: Latest stable version
* **Python**: Version 3.10 or higher
* **Libraries**: `polars`, `numpy`

### Installation & Execution

1. **Clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/GCP_FILES_FHE_IEEE_RUST.git
cd GCP_FILES_FHE_IEEE_RUST

```

2. **Install Python dependencies:**

```bash
pip install polars numpy

```

3. **Generate Validation Data:**
This script creates the required `.parquet` file for the Rust engine to read.

```bash
python3 scripts/generate_test_data.py

```

4. **Run the FHE Validation Pyramid:**

```bash
cargo run --release

```

## 📊 Performance Notes

* **Local (Apple Silicon):** Optimized for M-series chips using `aarch64` hardware acceleration.
* **Production (GCP):** Designed for `n2-standard-16` instances to utilize AVX-512 instructions.

## 📜 License

This project is open-source.

---

### **Instructions:**

1. Open your `README.md` file.
2. **Delete everything** inside it.
3. **Paste** the entire block above into the file.
4. **Save** the file.


import json
import os
import sys

# PATH CONFIGURATION
# We use relative paths to make this work on any machine (Mac, Linux, Windows)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, "data", "LR_weights_quantized.json")

def validate_weights():
    print(f"🔍 [VALIDATOR] Checking Model Weights at: {JSON_PATH}")

    # 1. Check File Existence
    if not os.path.exists(JSON_PATH):
        print(f"❌ ERROR: File not found at {JSON_PATH}")
        print("   -> Did you forget to copy the JSON file from the training artifacts?")
        sys.exit(1)

    # 2. Check JSON Structure
    try:
        with open(JSON_PATH, "r") as f:
            data = json.load(f)
        
        # Rust expects "bias" (int64) and "weights" (Vec<int64>)
        required_keys = ["bias", "weights"]
        for key in required_keys:
            if key not in data:
                print(f"❌ ERROR: Missing key '{key}' in JSON.")
                sys.exit(1)
        
        # 3. Check Data Types (Must be Integers for FHE)
        if not isinstance(data["bias"], int):
            print(f"❌ ERROR: 'bias' must be an integer, found {type(data['bias'])}")
            sys.exit(1)
            
        if not isinstance(data["weights"], list) or not all(isinstance(x, int) for x in data["weights"]):
            print("❌ ERROR: 'weights' must be a list of integers.")
            sys.exit(1)

        print(f"✅ SUCCESS: Model Weights are valid.")
        print(f"   -> Bias: {data['bias']}")
        print(f"   -> Features: {len(data['weights'])}")
        print("   -> Format matches Rust FHE Server requirements.")
        
    except json.JSONDecodeError:
        print("❌ ERROR: File is not valid JSON.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Unexpected failure: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    validate_weights()

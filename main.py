import os
import pandas as pd
import logging  # Added for logging
from eda import run_eda
from feature_engineering import run_feature_engineering
from partitioning import run_partitioning
from gen_data import main as generate_data
from utils import load_data, log_to_global_registry, make_param_hash
from datetime import datetime, timezone
from numeric_conversion import run_numeric_conversion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_config(config: dict) -> None:
    """Validate the configuration dictionary."""
    required_keys = [
        "target_col", "id_col", "seed", "train_size", "val_size", "test_size",
        "stratify_cardinality_threshold", "c1", "c2", "b1", "c3", "id_like_exempt"
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    if not (0 < config["train_size"] < 1 and 0 < config["val_size"] < 1 and 0 < config["test_size"] < 1):
        raise ValueError("Train, validation, and test sizes must be between 0 and 1.")
    if config["train_size"] + config["val_size"] + config["test_size"] != 1:
        raise ValueError("Train, validation, and test sizes must sum to 1.")

def create_directories(directories: list) -> None:
    """Create a list of directories if they do not exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_pipeline(config: dict, global_hash: str) -> None:
    try:
        validate_config(config)  # Validate config at the start
        create_directories(["artifacts", "artifacts/eda", "artifacts/step1", "artifacts/step2"])
        logging.info("Starting pipeline execution...")

        # Measure execution time for each step
        import time
        start_time = time.time()

        # Check if the database exists, if not, generate synthetic data
        db_path = "fraud_poc.db"
        if not os.path.exists(db_path):
            logging.info("[main] Database not found. Generating synthetic data...")
            generate_data()
        else:
            logging.info("[main] Database found. Skipping data generation.")

        # Load data
        df = load_data(db_path)

        # Step 0: EDA
        eda_config = {
            "target_col": config["target_col"],
            "columns": sorted(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns}
        }
        eda_hash = make_param_hash({**config, **eda_config})  # Combine global config with step-specific config

        # Log execution time for EDA
        eda_start = time.time()
        eda_path = run_eda(
            df=df,
            target_col=eda_config["target_col"],
            param_hash=eda_hash,
            use_mlflow=True,
            config=eda_config,
            # output_path=config.get("output_path", "artifacts/eda"),  # Optional output path
        )
        logging.info(f"EDA completed in {time.time() - eda_start:.2f} seconds.")

        log_to_global_registry({
            "step": "eda",
            "hash": eda_hash,
            "global_hash": global_hash,  # Include global hash for traceability
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": eda_config,
            "output_dir": eda_path
        })
        logging.info(f"[main] EDA artifacts saved at: {os.path.join('artifacts', 'eda')}")

        # Step 1: Feature Engineering
        step1_config = {
            "columns": sorted(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns}
        }
        step1_hash = make_param_hash({**config, **step1_config})  # Combine global config with step-specific config

        # Log execution time for Feature Engineering
        fe_start = time.time()
        df_fe, step1_path = run_feature_engineering(
            df=df,
            param_hash=step1_hash,
            use_mlflow=True,
            config=step1_config,
            # output_path=config.get("output_path", "artifacts/step1"),  # Optional output path
        )
        logging.info(f"Feature Engineering completed in {time.time() - fe_start:.2f} seconds.")

        log_to_global_registry({
            "step": "feature_engineering",
            "hash": step1_hash,
            "global_hash": global_hash,  # Include global hash for traceability
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": step1_config,
            "output_dir": step1_path
        })
        logging.info(f"[main] Feature Engineering artifacts saved at: {step1_path}")

        # Step 2: Partitioning
        step2_config = {
            "target_col": config["target_col"],
            "id_col": config["id_col"],
            "seed": config["seed"],
            "train_size": config["train_size"],
            "val_size": config["val_size"],
            "test_size": config["test_size"],
            "stratify_cardinality_threshold": config["stratify_cardinality_threshold"]
        }
        step2_hash = make_param_hash({**config, **step2_config})  # Combine global config with step-specific config

        # Log execution time for Partitioning
        partitioning_start = time.time()
        splits, step2_path = run_partitioning(
            df=df_fe,
            id_col=step2_config["id_col"],
            target_col=step2_config["target_col"],
            param_hash=step2_hash,
            seed=step2_config["seed"],
            train_size=step2_config["train_size"],
            val_size=step2_config["val_size"],
            test_size=step2_config["test_size"],
            stratify_cardinality_threshold=step2_config["stratify_cardinality_threshold"],
            use_mlflow=True
        )
        logging.info(f"Partitioning completed in {time.time() - partitioning_start:.2f} seconds.")

        log_to_global_registry({
            "step": "partitioning",
            "hash": step2_hash,
            "global_hash": global_hash,  # Include global hash for traceability
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": step2_config,
            "output_dir": step2_path
        })
        logging.info(f"[main] Partitioning artifacts saved at: {step2_path}")
        logging.info(f"Data partitioning completed. Partitioned data saved to: {step2_path}")

        # === STEP 3: Numeric Conversion + One-Hot Encoding ===
        step3_config = {
            "target_col": config["target_col"],
            "c1": 10,
            "c2": 0.01,
            "b1": True,
            "c3": 10,
            "id_like_exempt": True
        }
        step3_hash = make_param_hash(step3_config)

        # Log execution time for Numeric Conversion
        numeric_conversion_start = time.time()
        df_numeric, step3_path = run_numeric_conversion(
            df=df_fe,
            target_col=step3_config["target_col"],
            param_hash=step3_hash,
            config=step3_config,
            use_mlflow=True
        )
        logging.info(f"Numeric Conversion completed in {time.time() - numeric_conversion_start:.2f} seconds.")

        log_to_global_registry({
            "step": "numeric_conversion",
            "hash": step3_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": step3_config,
            "output_dir": step3_path
        })

        logging.info(f"[main] Numeric conversion saved at: {step3_path}")
        logging.info(f"Pipeline execution completed in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Define the global configuration for the entire pipeline
    config = {
        "target_col": "is_fraud",
        "id_col": "transaction_id",
        "seed": 42,
        "train_size": 0.6,
        "val_size": 0.1,
        "test_size": 0.3,
        "stratify_cardinality_threshold": 10,
        "c1": 10,
        "c2": 0.01,
        "b1": True,
        "c3": 10,
        "id_like_exempt": True
    }

    # Compute the global hash for the entire pipeline
    global_hash = make_param_hash(config)

    # Run the pipeline
    run_pipeline(config, global_hash)
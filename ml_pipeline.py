import os
import json
import logging
from datetime import datetime, timezone
import pandas as pd
import sqlite3
from ml_pipeline.eda import run_eda, make_param_hash as make_eda_hash
from ml_pipeline.feature_engineering import run_feature_engineering, make_param_hash as make_step1_hash
from ml_pipeline.partitioning import run_partitioning, make_param_hash as make_step2_hash
from ml_pipeline.numeric_conversion import run_numeric_conversion
from ml_pipeline.utils import load_data, log_to_global_registry, make_param_hash  # Fix: Import make_param_hash from utils
from gen_data import main as generate_data
from ml_pipeline.base import MLPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ...existing code removed...

def validate_config(config: dict):
    """Validate the configuration dictionary."""
    required_keys = [
        "target_column", "id_col", "seed", "train_size", "val_size", "test_size",
        "stratify_cardinality_threshold", "c1", "c2", "b1", "c3", "id_like_exempt"
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    if not (0 < config["train_size"] < 1 and 0 < config["val_size"] < 1 and 0 < config["test_size"] < 1):
        raise ValueError("Train, validation, and test sizes must be between 0 and 1.")
    if config["train_size"] + config["val_size"] + config["test_size"] != 1:
        raise ValueError("Train, validation, and test sizes must sum to 1.")


def create_directories():
    """Create necessary directories for artifacts."""
    directories = ["artifacts", "artifacts/eda", "artifacts/step1", "artifacts/step2", "artifacts/step3"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def log_registry(step: str, param_hash: str, config: dict, output_dir: str) -> None:
    """Log step details to the global registry."""
    os.makedirs(os.path.dirname("artifacts/global_registry.jsonl"), exist_ok=True)
    with open("artifacts/global_registry.jsonl", "a") as f:
        f.write(json.dumps({
            "step": step,
            "param_hash": param_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "output_dir": output_dir
        }) + "\n")


def load_data(db_path: str = "fraud_poc.db") -> pd.DataFrame:
    """Load data from the database."""
    conn = sqlite3.connect(db_path)
    df_clients = pd.read_sql_query("SELECT * FROM clients", conn)
    df_merchants = pd.read_sql_query("SELECT * FROM merchants", conn)
    df_transactions = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    df_clients.rename(columns={"account_creation_date": "account_creation_date_client"}, inplace=True)
    df_merchants.rename(columns={"account_creation_date": "account_creation_date_merchant"}, inplace=True)
    df = df_transactions.merge(df_clients, on="client_id").merge(df_merchants, on="merchant_id")
    return df


def run_step_0_eda(config: dict, dataframes: dict, hashes: dict, paths: dict):
    """Run Step 0: EDA."""
    df_raw = load_data()
    dataframes["raw"] = df_raw
    eda_config = {
        "target_column": config["target_column"],
        "columns": sorted(df_raw.columns),
        "dtypes": {col: str(df_raw[col].dtype) for col in df_raw.columns}
    }
    hash_id = make_eda_hash(eda_config)
    path = run_eda(df_raw, param_hash=hash_id, config=eda_config, use_mlflow=True)
    hashes["eda"] = hash_id
    paths["eda"] = path
    log_registry("eda", hash_id, eda_config, path)
    logging.info(f"[EDA] Artifacts saved at: {path}")


def run_step_1_feature_engineering(config: dict, dataframes: dict, hashes: dict, paths: dict):
    """Run Step 1: Feature Engineering."""
    df_raw = dataframes["raw"]
    fe_config = {
        "columns": sorted(df_raw.columns),
        "dtypes": {col: str(df_raw[col].dtype) for col in df_raw.columns}
    }
    hash_id = make_step1_hash(fe_config)
    df_fe, path = run_feature_engineering(df_raw, param_hash=hash_id, config=fe_config, use_mlflow=True)
    dataframes["feature_engineered"] = df_fe
    hashes["feature_engineering"] = hash_id
    paths["feature_engineering"] = path
    log_registry("feature_engineering", hash_id, fe_config, path)
    logging.info(f"[Feature Engineering] Artifacts saved at: {path}")


def run_step_2_partitioning(config: dict, dataframes: dict, hashes: dict, paths: dict):
    """Run Step 2: Partitioning."""
    df = dataframes["feature_engineered"]
    partitioning_config = {
        "target_column": config["target_column"],
        "id_col": config["id_col"],
        "seed": config["seed"],
        "train_size": config["train_size"],
        "val_size": config["val_size"],
        "test_size": config["test_size"],
        "stratify_cardinality_threshold": config["stratify_cardinality_threshold"]
    }
    hash_id = make_step2_hash(partitioning_config)
    splits, path = run_partitioning(
        df=df,
        id_col=partitioning_config["id_col"],
        target_col=partitioning_config["target_column"],
        param_hash=hash_id,
        use_mlflow=True
    )
    dataframes.update(splits)
    hashes["partitioning"] = hash_id
    paths["partitioning"] = path
    log_registry("partitioning", hash_id, partitioning_config, path)
    logging.info(f"[Partitioning] Artifacts saved at: {path}")


def run_step_3_numeric_conversion(config: dict, dataframes: dict, hashes: dict, paths: dict):
    """Run Step 3: Numeric Conversion + One-Hot Encoding."""
    df = dataframes["feature_engineered"]
    numeric_config = {
        "target_column": config["target_column"],
        "c1": config["c1"],
        "c2": config["c2"],
        "b1": config["b1"],
        "c3": config["c3"],
        "id_like_exempt": config["id_like_exempt"]
    }
    step3_hash = make_param_hash(numeric_config)  # Fix: Use make_param_hash from utils
    df_numeric, path = run_numeric_conversion(
        df=df,
        target_col=numeric_config["target_column"],
        param_hash=step3_hash,
        config=numeric_config,
        use_mlflow=True
    )
    dataframes["numeric"] = df_numeric
    hashes["numeric_conversion"] = step3_hash
    paths["numeric_conversion"] = path
    log_registry("numeric_conversion", step3_hash, numeric_config, path)
    logging.info(f"[Numeric Conversion] Artifacts saved at: {path}")


def run_all(config: dict):
    """Run all pipeline steps."""
    validate_config(config)
    create_directories()

    dataframes = {}
    hashes = {}
    paths = {}

    # Check if the database exists, if not, generate synthetic data
    db_path = "fraud_poc.db"
    if not os.path.exists(db_path):
        logging.info("[Pipeline] Database not found. Generating synthetic data...")
        generate_data()
    else:
        logging.info("[Pipeline] Database found. Skipping data generation.")

    # Load data
    dataframes["raw"] = load_data(db_path)

    # Run pipeline steps
    run_step_0_eda(config, dataframes, hashes, paths)
    run_step_1_feature_engineering(config, dataframes, hashes, paths)
    run_step_2_partitioning(config, dataframes, hashes, paths)
    run_step_3_numeric_conversion(config, dataframes, hashes, paths)
    logging.info("[Pipeline] All steps completed successfully.")
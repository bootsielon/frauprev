import os
import pandas as pd
from eda import run_eda
from feature_engineering import run_feature_engineering
from partitioning import run_partitioning
from gen_data import main as generate_data
from utils import load_data, log_to_global_registry, make_param_hash
from datetime import datetime, timezone


def run_pipeline(config: dict, global_hash: str) -> None:
    # Ensure artifact directories exist
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("artifacts/eda", exist_ok=True)
    os.makedirs("artifacts/step1", exist_ok=True)
    os.makedirs("artifacts/step2", exist_ok=True)

    # Check if the database exists, if not, generate synthetic data
    db_path = "fraud_poc.db"
    if not os.path.exists(db_path):
        print("[main] Database not found. Generating synthetic data...")
        generate_data()
    else:
        print("[main] Database found. Skipping data generation.")

    # Load data
    df = load_data(db_path)

    # Step 0: EDA
    eda_config = {
        "target_col": config["target_col"],
        "columns": sorted(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }
    eda_hash = make_param_hash({**config, **eda_config})  # Combine global config with step-specific config
    eda_path = run_eda(
        df=df,
        target_col=eda_config["target_col"],
        param_hash=eda_hash,
        use_mlflow=True,
        config=eda_config,
        # output_path="artifacts/eda",
    )
    log_to_global_registry({
        "step": "eda",
        "hash": eda_hash,
        "global_hash": global_hash,  # Include global hash for traceability
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": eda_config,
        "output_dir": eda_path
    })
    print(f"[main] EDA artifacts saved at: {os.path.join('artifacts', 'eda')}")

    # Step 1: Feature Engineering
    step1_config = {
        "columns": sorted(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }
    step1_hash = make_param_hash({**config, **step1_config})  # Combine global config with step-specific config
    df_fe, step1_path = run_feature_engineering(
        df=df,
        param_hash=step1_hash,
        use_mlflow=True,
        config=step1_config,
        # output_path="artifacts/step1",
    )
    log_to_global_registry({
        "step": "feature_engineering",
        "hash": step1_hash,
        "global_hash": global_hash,  # Include global hash for traceability
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": step1_config,
        "output_dir": step1_path
    })
    print(f"[main] Feature Engineering artifacts saved at: {step1_path}")

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
    log_to_global_registry({
        "step": "partitioning",
        "hash": step2_hash,
        "global_hash": global_hash,  # Include global hash for traceability
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": step2_config,
        "output_dir": step2_path
    })
    print(f"[main] Partitioning artifacts saved at: {step2_path}")
    print("Data partitioning completed. Partitioned data saved to:", step2_path)


if __name__ == "__main__":
    # Define the global configuration for the entire pipeline
    config = {
        "target_col": "is_fraud",
        "id_col": "transaction_id",
        "seed": 42,
        "train_size": 0.6,
        "val_size": 0.1,
        "test_size": 0.3,
        "stratify_cardinality_threshold": 10
    }

    # Compute the global hash for the entire pipeline
    global_hash = make_param_hash(config)

    # Run the pipeline
    run_pipeline(config, global_hash)
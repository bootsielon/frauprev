import os
import json
import logging
from datetime import datetime, timezone
import pandas as pd
import sqlite3
from eda import run_eda, make_param_hash as make_eda_hash
from feature_engineering import run_feature_engineering, make_param_hash as make_step1_hash
from partitioning import run_partitioning, make_param_hash as make_step2_hash
from numeric_conversion import run_numeric_conversion
from utils import load_data, log_to_global_registry, make_param_hash  # Fix: Import make_param_hash from utils
from gen_data import main as generate_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MLPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.dataframes = {}
        self.paths = {}
        self.hashes = {}
        self.global_hash = make_param_hash(config)
        self.registry_path = "artifacts/global_registry.jsonl"
        os.makedirs("artifacts", exist_ok=True)

    def validate_config(self):
        """Validate the configuration dictionary."""
        required_keys = [
            "target_col", "id_col", "seed", "train_size", "val_size", "test_size",
            "stratify_cardinality_threshold", "c1", "c2", "b1", "c3", "id_like_exempt"
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        if not (0 < self.config["train_size"] < 1 and 0 < self.config["val_size"] < 1 and 0 < self.config["test_size"] < 1):
            raise ValueError("Train, validation, and test sizes must be between 0 and 1.")
        if self.config["train_size"] + self.config["val_size"] + self.config["test_size"] != 1:
            raise ValueError("Train, validation, and test sizes must sum to 1.")

    def create_directories(self):
        """Create necessary directories for artifacts."""
        directories = ["artifacts", "artifacts/eda", "artifacts/step1", "artifacts/step2", "artifacts/step3"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def log_registry(self, step: str, param_hash: str, config: dict, output_dir: str) -> None:
        """Log step details to the global registry."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "a") as f:
            f.write(json.dumps({
                "step": step,
                "param_hash": param_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "output_dir": output_dir
            }) + "\n")

    def load_data(self, db_path: str = "fraud_poc.db") -> pd.DataFrame:
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

    def run_step_0_eda(self):
        """Run Step 0: EDA."""
        df_raw = self.load_data()
        self.dataframes["raw"] = df_raw
        config = {
            "target_col": self.config["target_col"],
            "columns": sorted(df_raw.columns),
            "dtypes": {col: str(df_raw[col].dtype) for col in df_raw.columns}
        }
        hash_id = make_eda_hash(config)
        path = run_eda(df_raw, target_col=config["target_col"], param_hash=hash_id, config=config, use_mlflow=True)
        self.hashes["eda"] = hash_id
        self.paths["eda"] = path
        self.log_registry("eda", hash_id, config, path)
        logging.info(f"[EDA] Artifacts saved at: {path}")

    def run_step_1_feature_engineering(self):
        """Run Step 1: Feature Engineering."""
        df_raw = self.dataframes["raw"]
        config = {
            "columns": sorted(df_raw.columns),
            "dtypes": {col: str(df_raw[col].dtype) for col in df_raw.columns}
        }
        hash_id = make_step1_hash(config)
        df_fe, path = run_feature_engineering(df_raw, param_hash=hash_id, config=config, use_mlflow=True)
        self.dataframes["feature_engineered"] = df_fe
        self.hashes["feature_engineering"] = hash_id
        self.paths["feature_engineering"] = path
        self.log_registry("feature_engineering", hash_id, config, path)
        logging.info(f"[Feature Engineering] Artifacts saved at: {path}")

    def run_step_2_partitioning(self):
        """Run Step 2: Partitioning."""
        df = self.dataframes["feature_engineered"]
        config = {
            "target_col": self.config["target_col"],
            "id_col": self.config["id_col"],
            "seed": self.config["seed"],
            "train_size": self.config["train_size"],
            "val_size": self.config["val_size"],
            "test_size": self.config["test_size"],
            "stratify_cardinality_threshold": self.config["stratify_cardinality_threshold"]
        }
        hash_id = make_step2_hash(config)
        splits, path = run_partitioning(
            df=df,
            id_col=config["id_col"],
            target_col=config["target_col"],
            param_hash=hash_id,
            #config=config,
            use_mlflow=True
        )
        self.dataframes.update(splits)
        self.hashes["partitioning"] = hash_id
        self.paths["partitioning"] = path
        self.log_registry("partitioning", hash_id, config, path)
        logging.info(f"[Partitioning] Artifacts saved at: {path}")

    def run_step_3_numeric_conversion(self):
        """Run Step 3: Numeric Conversion + One-Hot Encoding."""
        df = self.dataframes["feature_engineered"]
        config = {
            "target_col": self.config["target_col"],
            "c1": self.config["c1"],
            "c2": self.config["c2"],
            "b1": self.config["b1"],
            "c3": self.config["c3"],
            "id_like_exempt": self.config["id_like_exempt"]
        }
        step3_hash = make_param_hash(config)  # Fix: Use make_param_hash from utils
        df_numeric, path = run_numeric_conversion(
            df=df,
            target_col=config["target_col"],
            param_hash=step3_hash,
            config=config,
            use_mlflow=True
        )
        self.dataframes["numeric"] = df_numeric
        self.hashes["numeric_conversion"] = step3_hash
        self.paths["numeric_conversion"] = path
        self.log_registry("numeric_conversion", step3_hash, config, path)
        logging.info(f"[Numeric Conversion] Artifacts saved at: {path}")

    def run_all(self):
        """Run all pipeline steps."""
        self.validate_config()
        self.create_directories()

        # Check if the database exists, if not, generate synthetic data
        db_path = "fraud_poc.db"
        if not os.path.exists(db_path):
            logging.info("[Pipeline] Database not found. Generating synthetic data...")
            generate_data()
        else:
            logging.info("[Pipeline] Database found. Skipping data generation.")

        # Load data
        self.dataframes["raw"] = load_data(db_path)

        # Run pipeline steps
        self.run_step_0_eda()
        self.run_step_1_feature_engineering()
        self.run_step_2_partitioning()
        self.run_step_3_numeric_conversion()
        logging.info("[Pipeline] All steps completed successfully.")
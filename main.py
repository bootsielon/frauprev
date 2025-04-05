import os
import pandas as pd
import logging
from eda import run_eda
from feature_engineering import run_feature_engineering
from partitioning import run_partitioning
from numeric_conversion import run_numeric_conversion
from gen_data import main as generate_data
from utils import load_data, log_to_global_registry, make_param_hash
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FraudPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.hashes = {}
        self.paths = {}
        self.dataframes = {}
        self.global_hash = make_param_hash(config)

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

    def run_step_0_eda(self):
        """Run Step 0: EDA."""
        df = self.dataframes.get("raw")
        eda_config = {
            "target_col": self.config["target_col"],
            "columns": sorted(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns}
        }
        eda_hash = make_param_hash({**self.config, **eda_config})
        eda_path = run_eda(
            df=df,
            target_col=eda_config["target_col"],
            param_hash=eda_hash,
            use_mlflow=True,
            config=eda_config
        )
        self.hashes["eda"] = eda_hash
        self.paths["eda"] = eda_path
        log_to_global_registry({
            "step": "eda",
            "hash": eda_hash,
            "global_hash": self.global_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": eda_config,
            "output_dir": eda_path
        })
        logging.info(f"[EDA] Artifacts saved at: {eda_path}")

    def run_step_1_feature_engineering(self):
        """Run Step 1: Feature Engineering."""
        df = self.dataframes.get("raw")
        step1_config = {
            "columns": sorted(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns}
        }
        step1_hash = make_param_hash({**self.config, **step1_config})
        df_fe, step1_path = run_feature_engineering(
            df=df,
            param_hash=step1_hash,
            use_mlflow=True,
            config=step1_config
        )
        self.dataframes["feature_engineered"] = df_fe
        self.hashes["feature_engineering"] = step1_hash
        self.paths["feature_engineering"] = step1_path
        log_to_global_registry({
            "step": "feature_engineering",
            "hash": step1_hash,
            "global_hash": self.global_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": step1_config,
            "output_dir": step1_path
        })
        logging.info(f"[Feature Engineering] Artifacts saved at: {step1_path}")

    def run_step_2_partitioning(self):
        """Run Step 2: Partitioning."""
        df = self.dataframes.get("feature_engineered")
        step2_config = {
            "target_col": self.config["target_col"],
            "id_col": self.config["id_col"],
            "seed": self.config["seed"],
            "train_size": self.config["train_size"],
            "val_size": self.config["val_size"],
            "test_size": self.config["test_size"],
            "stratify_cardinality_threshold": self.config["stratify_cardinality_threshold"]
        }
        step2_hash = make_param_hash({**self.config, **step2_config})
        splits, step2_path = run_partitioning(
            df=df,
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
        self.dataframes.update(splits)
        self.hashes["partitioning"] = step2_hash
        self.paths["partitioning"] = step2_path
        log_to_global_registry({
            "step": "partitioning",
            "hash": step2_hash,
            "global_hash": self.global_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": step2_config,
            "output_dir": step2_path
        })
        logging.info(f"[Partitioning] Artifacts saved at: {step2_path}")

    def run_step_3_numeric_conversion(self):
        """Run Step 3: Numeric Conversion + One-Hot Encoding."""
        df = self.dataframes.get("feature_engineered")
        step3_config = {
            "target_col": self.config["target_col"],
            "c1": self.config["c1"],
            "c2": self.config["c2"],
            "b1": self.config["b1"],
            "c3": self.config["c3"],
            "id_like_exempt": self.config["id_like_exempt"]
        }
        step3_hash = make_param_hash(step3_config)
        df_numeric, step3_path = run_numeric_conversion(
            df=df,
            target_col=step3_config["target_col"],
            param_hash=step3_hash,
            config=step3_config,
            use_mlflow=True
        )
        self.dataframes["numeric"] = df_numeric
        self.hashes["numeric_conversion"] = step3_hash
        self.paths["numeric_conversion"] = step3_path
        log_to_global_registry({
            "step": "numeric_conversion",
            "hash": step3_hash,
            "global_hash": self.global_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": step3_config,
            "output_dir": step3_path
        })
        logging.info(f"[Numeric Conversion] Artifacts saved at: {step3_path}")

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

    # Run the pipeline
    pipeline = FraudPipeline(config)
    pipeline.run_all()
import pandas as pd
from .eda import eda
from .feature_engineering import run_feature_engineering
from .partitioning import run_partitioning
from .numeric_conversion import run_numeric_conversion
from utils import make_param_hash  # Add this import

class MLPipeline:
    """
    Master orchestration object for configuration-driven ML pipeline execution.
    Stores full state, supports hash-based tracking, checkpointing, and recovery.
    """
    def __init__(self, config: dict):
        self.config = config
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.paths: dict[str, str] = {}
        self.hashes: dict[str, str] = {}

    def load_data(self, db_path: str = "fraud_poc.db") -> pd.DataFrame:
        # ...existing code...
        import sqlite3
        conn = sqlite3.connect(db_path)
        df_clients = pd.read_sql("SELECT * FROM clients", conn)
        df_merchants = pd.read_sql("SELECT * FROM merchants", conn)
        df_tx = pd.read_sql("SELECT * FROM transactions", conn)
        conn.close()
        df_clients.rename(columns={"account_creation_date": "account_creation_date_client"}, inplace=True)
        df_merchants.rename(columns={"account_creation_date": "account_creation_date_merchant"}, inplace=True)
        merged_df = df_tx.merge(df_clients, on="client_id").merge(df_merchants, on="merchant_id")
        # Store in the dataframes dictionary
        self.dataframes["original"] = merged_df
        # Also store as "cleaned" since we might not have a cleaning step
        self.dataframes["cleaned"] = merged_df.copy()
        print(f"[INFO] Data loaded with {len(merged_df)} rows and {merged_df.shape[1]} columns")
        return merged_df

    def run_all(self) -> None:
        # Check if data is loaded, if not, load it
        if not self.dataframes or "original" not in self.dataframes:
            print("[INFO] No data found in pipeline, loading data...")
            self.load_data()
        
        print("[INFO] Running EDA step...")
        eda(self)
        
        print("[INFO] Running feature engineering step...")
        try:
            run_feature_engineering(self, self.hashes, self.config)  # Updated call with required arguments
        except Exception as e:
            print(f"[WARN] Feature engineering error: {e}. Using cleaned data instead.")
            if "cleaned" in self.dataframes:
                self.dataframes["feature_engineered"] = self.dataframes["cleaned"].copy()
            else:
                self.dataframes["feature_engineered"] = self.dataframes["original"].copy()
                self.dataframes["cleaned"] = self.dataframes["original"].copy()
        
        print("[INFO] Running partitioning step...")
        run_partitioning_from_pipeline(self)  # Modified: use wrapper function
        
        print("[INFO] Running numeric conversion step...")
        run_numeric_conversion_from_pipeline(self)  # Modified: use wrapper function

def run_partitioning_from_pipeline(pipeline):
    """Wrapper function to run partitioning from pipeline object"""
    
    # Debug the available dataframes
    print(f"[DEBUG] Available dataframes: {list(pipeline.dataframes.keys())}")
    
    # Check if feature_engineered exists, otherwise use cleaned or original data
    if "feature_engineered" in pipeline.dataframes:
        df_to_use = pipeline.dataframes["feature_engineered"]
        print(f"[INFO] Using 'feature_engineered' dataframe for partitioning with shape {df_to_use.shape}")
    elif "cleaned" in pipeline.dataframes:
        df_to_use = pipeline.dataframes["cleaned"]
        print(f"[WARN] 'feature_engineered' dataframe not found, using 'cleaned' instead with shape {df_to_use.shape}")
    elif "original" in pipeline.dataframes:
        df_to_use = pipeline.dataframes["original"]
        print(f"[WARN] 'feature_engineered' and 'cleaned' dataframes not found, using 'original' instead with shape {df_to_use.shape}")
    else:
        # If we still don't have data, try loading it
        try:
            print("[WARN] No dataframes found. Attempting to load data...")
            pipeline.load_data()
            if "original" in pipeline.dataframes:
                df_to_use = pipeline.dataframes["original"]
                print(f"[INFO] Using freshly loaded 'original' data with shape {df_to_use.shape}")
            else:
                raise KeyError("Failed to load data")
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            raise KeyError("No suitable dataframe found in pipeline for partitioning")
    
    try:
        # Generate param_hash for partitioning
        partitioning_config = {
            "id_col": pipeline.config.get("id_col", "transaction_id"),
            "target_col": pipeline.config.get("target_column", "is_fraud"),
            "seed": pipeline.config.get("seed", 42),
            "train_size": pipeline.config.get("train_size", 0.6),
            "val_size": pipeline.config.get("val_size", 0.1),
            "test_size": pipeline.config.get("test_size", 0.3),
            "stratify_cardinality_threshold": pipeline.config.get("stratify_cardinality_threshold", 10)
        }
        param_hash = make_param_hash(partitioning_config)
        
        # Run partitioning with proper arguments
        splits, artifact_path = run_partitioning(
            df=df_to_use,
            id_col=partitioning_config["id_col"],
            target_col=partitioning_config["target_col"],
            param_hash=param_hash,
            output_path="artifacts/step2",
            seed=partitioning_config["seed"],
            train_size=partitioning_config["train_size"],
            val_size=partitioning_config["val_size"],
            test_size=partitioning_config["test_size"],
            stratify_cardinality_threshold=partitioning_config["stratify_cardinality_threshold"],
            use_mlflow=True
        )
        
        # Store splits in the pipeline's dataframes dictionary
        pipeline.dataframes.update(splits)
        pipeline.paths["partitioning"] = artifact_path
        pipeline.hashes["partitioning"] = param_hash
        
        print(f"[INFO] Data partitioning completed successfully. Splits: {list(splits.keys())}")
    except Exception as e:
        print(f"[ERROR] Error during partitioning: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_numeric_conversion_from_pipeline(pipeline):
    """Wrapper function to run numeric conversion from pipeline object"""
    
    # Debug the available dataframes
    print(f"[DEBUG] Available dataframes for numeric conversion: {list(pipeline.dataframes.keys())}")
    
    # Check which dataframe to use
    if "train" in pipeline.dataframes:
        df_to_use = pipeline.dataframes["train"]
        print(f"[INFO] Using 'train' dataframe for numeric conversion with shape {df_to_use.shape}")
    elif "feature_engineered" in pipeline.dataframes:
        df_to_use = pipeline.dataframes["feature_engineered"]
        print(f"[WARN] 'train' dataframe not found, using 'feature_engineered' instead with shape {df_to_use.shape}")
    elif "cleaned" in pipeline.dataframes:
        df_to_use = pipeline.dataframes["cleaned"]
        print(f"[WARN] 'train' and 'feature_engineered' dataframes not found, using 'cleaned' instead with shape {df_to_use.shape}")
    elif "original" in pipeline.dataframes:
        df_to_use = pipeline.dataframes["original"]
        print(f"[WARN] Using 'original' dataframe for numeric conversion with shape {df_to_use.shape}")
    else:
        raise KeyError("No suitable dataframe found in pipeline for numeric conversion")
    
    try:
        # Generate numeric conversion config
        numeric_config = {
            "target_col": pipeline.config.get("target_column", "is_fraud"),
            "c1": pipeline.config.get("c1", 0.95),
            "c2": pipeline.config.get("c2", 0.95),
            "b1": pipeline.config.get("b1", 10),
            "c3": pipeline.config.get("c3", 0.95),
            "id_like_exempt": pipeline.config.get("id_like_exempt", ["transaction_id", "client_id", "merchant_id"])
        }
        
        param_hash = make_param_hash(numeric_config)
        
        # Run numeric conversion with proper arguments
        df_numeric, artifact_path = run_numeric_conversion(
            df=df_to_use,
            target_col=numeric_config["target_col"],
            param_hash=param_hash,
            config=numeric_config,
            use_mlflow=True
        )
        
        # Store the result in the pipeline's dataframes dictionary
        pipeline.dataframes["numeric"] = df_numeric
        pipeline.paths["numeric_conversion"] = artifact_path
        pipeline.hashes["numeric_conversion"] = param_hash
        
        print(f"[INFO] Numeric conversion completed successfully with shape {df_numeric.shape}")
    except Exception as e:
        print(f"[ERROR] Error during numeric conversion: {e}")
        import traceback
        traceback.print_exc()
        raise

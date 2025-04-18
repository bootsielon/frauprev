import os
import json
from datetime import datetime
import pandas as pd
# import numpy as np
import mlflow
from ml_pipeline.utils import make_param_hash, log_registry, convert_numpy_types


def feature_engineering(self) -> None:
    """
    Step 1: Add derived features, drop zero-variance or constant columns.
    Updates: self.dataframes["feature_engineered"], self.paths["feature_engineering"], self.hashes["feature_engineering"]
    """
    df = self.dataframes["raw"]
    global_train_hash = self.config.get("global_train_hash")
    step = "feature_engineering"
    train_mode = self.train_mode 
    config = {
        "columns": sorted(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "n_missing": df.isnull().sum().to_dict(),
        "n_unique": df.nunique().to_dict(),
        "n_duplicates": df.duplicated().sum(),
        "n_constant": df.nunique(dropna=False).eq(1).sum(),
        "n_zero_variance": df.nunique(dropna=False).eq(1).sum(),
        "n_constant_columns": df.nunique(dropna=False).eq(1).sum(),
        "n_zero_variance_columns": df.nunique(dropna=False).eq(1).sum(),
        "n_categorical": df.select_dtypes(include=["object", "category"]).shape[1],
        "n_numeric": df.select_dtypes(include=["number"]).shape[1],
        "train_mode": self.train_mode,
        "target_col": self.config.get("target_col"),
        # "target_dtype": str(df[self.config["target_col"]].dtype) if self.config["target_col"] in df.columns else None,
    }
    
    param_hash = make_param_hash(config) if self.train_mode else self.config["train_hash"]  # param_hash = make_param_hash(config) 
    step_dir = os.path.join("artifacts", f"{step}_{param_hash}")
    
    manifest_file = os.path.join(step_dir, "manifest.json")
    inf_hash = make_param_hash(config) if not self.train_mode else None
    inf_step_dir = os.path.join("artifacts", f"{step}_{inf_hash}")
    inf_manifest_file = os.path.join(inf_step_dir, "manifest.json")

    final_hash = param_hash if train_mode else inf_hash
    final_dir = step_dir if train_mode else inf_step_dir
    if not os.path.exists(final_dir):
        os.makedirs(final_dir, exist_ok=True)
    # Save the manifest file in the final directory

    if os.path.exists(manifest_file) and train_mode:
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = param_hash
        self.dataframes["feature_engineered"] = pd.read_csv(os.path.join(step_dir, f"{step}_{param_hash}.csv"))
        with open(manifest_file, "r") as f:
            manifest = json.load(f)
            self.artifacts[step] = manifest["outputs"]
        return
    elif os.path.exists(manifest_file) and not train_mode:
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = inf_hash
        with open(manifest_file, "r") as f:
            manifest = json.load(f)
            # remove the datasets from the manifest["outputs"]
            del manifest["outputs"]["engineered_csv"]
            self.artifacts[step] = manifest["outputs"]            
        if os.path.exists(inf_manifest_file):  
            # read the data for inference
            with open(inf_manifest_file, "r") as f:
                inf_manifest = json.load(f)            
                self.dataframes["feature_engineered"] = pd.read_csv(os.path.join(inf_step_dir, f"{step}_{inf_hash}.csv"))
                inf_manifest["outputs"]["engineered_csv"] = self.dataframes["feature_engineered"] 
                # combine manifest["outputs"] and inf_manifest["outputs"], make sure to remove datasets from manifest["outputs"] and only retain the datasets from inf_manifest["outputs"]
                inf_manifest["outputs"]["artifacts"] = manifest["outputs"]["artifacts"]
                self.artifacts[step] = inf_manifest  # {**self.artifacts[step], **inf_manifest["outputs"]}
                return
    elif not train_mode and not os.path.exists(manifest_file):  
        # fail gracefully if no manifest file exists for inference
        print(f"[{step.upper()}] Error: No manifest file found for inference at {step_dir}")
        print(f"Please ensure the model was trained and the manifest file exists at {step_dir}")
        # Fail gracefully
        assert False, f"[{step.upper()}] Error: No manifest file found for inference at {step_dir}"        

    if train_mode:
        os.makedirs(step_dir, exist_ok=True)
    else:
        os.makedirs(inf_step_dir, exist_ok=True)
        
    df_fe = df.copy()

    # Feature engineering example: timestamp-based
    if "timestamp" in df_fe.columns:
        df_fe["transaction_hour"] = pd.to_datetime(df_fe["timestamp"]).dt.hour
        df_fe["transaction_dayofweek"] = pd.to_datetime(df_fe["timestamp"]).dt.dayofweek

    for col in ["account_creation_date_client", "account_creation_date_merchant"]:
        if col in df_fe.columns:
            df_fe[col] = pd.to_datetime(df_fe[col], errors="coerce")
            df_fe[f"{col}_year"] = df_fe[col].dt.year
            df_fe[f"{col}_month"] = df_fe[col].dt.month
            df_fe[f"{col}_day"] = df_fe[col].dt.day
            df_fe[f"{col}_hour"] = df_fe[col].dt.hour
            df_fe[f"{col}_dayofweek"] = df_fe[col].dt.dayofweek
            if "timestamp" in df_fe.columns:
                df_fe[f"{col}_age_days"] = (pd.to_datetime(df_fe["timestamp"]) - df_fe[col]).dt.days
                df_fe[f"{col}_age_years"] = df_fe[f"{col}_age_days"] / 365.25


    #if "account_creation_date_client" in df_fe.columns:
        #df_fe["client_account_age_days"] = (
        #    pd.to_datetime(df_fe["timestamp"]) - pd.to_datetime(df_fe["account_creation_date_client"])
        #).dt.days

    #if "account_creation_date_merchant" in df_fe.columns:
        #df_fe["merchant_account_age_days"] = (
        #    pd.to_datetime(df_fe["timestamp"]) - pd.to_datetime(df_fe["account_creation_date_merchant"])
        #).dt.days

    # Drop constant columns (0 variance or same value)
    
    dropped = []

    if train_mode:
        for col in df_fe.columns:
            if df_fe[col].nunique(dropna=False) <= 1:
                dropped.append(col)
        df_fe.drop(columns=dropped, inplace=True)
    else:
        dropped = manifest["outputs"]["dropped_features"]
        # For inference, drop columns that are not in the training set
        df_fe.drop(columns=[col for col in dropped if col in df_fe.columns], inplace=True)

    output_csv = os.path.join(final_dir, f"{step}_{final_hash}.csv")

    drop_log = os.path.join(final_dir, f"dropped_features_{final_hash}.json") # if train_mode else os.path.join(inf_step_dir, f"dropped_features_{inf_hash}.json")

    df_fe.to_csv(output_csv, index=False)
    with open(drop_log, "w") as f:
        json.dump({"dropped_features": dropped}, f, indent=2)

    manifest = {
        "step": step,
        "param_hash": final_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": final_dir,
        "outputs": {
            "engineered_csv": output_csv,
            "dropped_features": dropped,
            "dropped_features_count": len(dropped),
            "dropped_features_log": drop_log,
            "retained_features": list(set(df_fe.columns) - set(dropped)),
            "initial_features": list(df.columns),
            "initial_shape": df.shape,
            "final_shape": df_fe.shape,
            "final_features": list(df_fe.columns),
            "new_features": list(set(df_fe.columns) - set(df.columns)),
        }
    }
    manifest = convert_numpy_types(manifest)  # Convert NumPy types to Python native types
    final_manifest_file = os.path.join(final_dir, "manifest.json")
    with open(final_manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{final_hash}"):
            mlflow.set_tags({"step": step, "param_hash": final_hash})
            mlflow.log_artifacts(final_dir, artifact_path=step)
    config = convert_numpy_types(config)  # Convert NumPy types to Python native types
    log_registry(step, param_hash, config, final_dir)
    self.dataframes["feature_engineered"] = df_fe
    self.paths[step] = final_dir
    self.hashes[step] = final_hash

if __name__ == "__main__":
    # Example usage to test the feature engineering functionality
    from ml_pipeline.base import MLPipeline
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create a simple mock dataset with timestamp and account creation dates
    now = datetime.now()
    mock_data = pd.DataFrame({
        "client_id": [1, 2, 3, 4, 5],
        "merchant_id": [101, 102, 103, 101, 102],
        "amount": [100.0, 50.0, 200.0, 75.0, 125.0],
        "is_fraud": [0, 0, 1, 0, 1],
        "timestamp": [
            now - timedelta(days=1),
            now - timedelta(days=2),
            now - timedelta(days=3),
            now - timedelta(days=4),
            now - timedelta(days=5)
        ],
        "account_creation_date_client": [
            now - timedelta(days=100),
            now - timedelta(days=200),
            now - timedelta(days=300),
            now - timedelta(days=400),
            now - timedelta(days=500)
        ],
        "account_creation_date_merchant": [
            now - timedelta(days=1000),
            now - timedelta(days=1200),
            now - timedelta(days=1300),
            now - timedelta(days=1400),
            now - timedelta(days=1500)
        ],
        "constant_col": [1, 1, 1, 1, 1]  # This should be dropped by feature engineering
    })
    
    # Convert datetime columns to string format similar to your database
    for dt_col in ['timestamp', 'account_creation_date_client', 'account_creation_date_merchant']:
        mock_data[dt_col] = mock_data[dt_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create a simple configuration for testing
    test_config = {
        "target_col": "is_fraud",
        "use_mlflow": False
    }
    
    # Initialize the pipeline with test configuration
    pipeline = MLPipeline(config=test_config)
    
    # Inject our mock data into the pipeline
    pipeline.dataframes["raw"] = mock_data
    
    # Run the feature engineering step
    pipeline.feature_engineering()
    
    # Check the results
    df_result = pipeline.dataframes["feature_engineered"]
    
    print("Feature Engineering completed successfully!")
    print(f"Original DataFrame shape: {mock_data.shape}")
    print(f"Engineered DataFrame shape: {df_result.shape}")
    print(f"New features added: {set(df_result.columns) - set(mock_data.columns)}")
    print(f"Features dropped: {set(mock_data.columns) - set(df_result.columns)}")
    print(f"Output directory: {pipeline.paths['feature_engineering']}")
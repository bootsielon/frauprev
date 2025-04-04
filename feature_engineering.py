import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timezone
import mlflow
from utils import make_param_hash, load_data

def add_derived_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Adds derived features to the DataFrame based on existing columns.

    Args:
        df (pd.DataFrame): Input DataFrame with transaction data.
    
    
    Returns:
        tuple: Transformed DataFrame and list of derived feature names.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["account_creation_date_client"] = pd.to_datetime(df["account_creation_date_client"])
    df["account_creation_date_merchant"] = pd.to_datetime(df["account_creation_date_merchant"])

    df["transaction_hour"] = df["timestamp"].dt.hour
    df["transaction_day"] = df["timestamp"].dt.dayofweek
    df["client_account_age_days"] = (df["timestamp"] - df["account_creation_date_client"]).dt.days
    df["merchant_account_age_days"] = (df["timestamp"] - df["account_creation_date_merchant"]).dt.days

    df.drop(columns=["timestamp", "account_creation_date_client", "account_creation_date_merchant"], inplace=True)
    return df, ["transaction_hour", "transaction_day", "client_account_age_days", "merchant_account_age_days"]

def drop_low_variance_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    zero_var = df.loc[:, df.nunique(dropna=False) <= 1].columns.tolist()
    df = df.drop(columns=zero_var)

    non_numeric_single_val = [
        col for col in df.select_dtypes(include="object").columns
        if df[col].nunique(dropna=False) == 1
    ]
    df = df.drop(columns=non_numeric_single_val)

    return df, {
        "zero_variance": zero_var,
        "non_numeric_single_value": non_numeric_single_val
    }


def run_feature_engineering(
    df: pd.DataFrame,
    param_hash: str,
    config: dict,
    output_dir: str = "artifacts/step1",
    use_mlflow: bool = False
) -> tuple[pd.DataFrame, str]:
    """
    Adds derived features and drops zero-variance and single-unique non-numeric columns.
    Uses hashing of df schema to cache outputs. Optionally logs artifacts to MLflow.

    Args:
        df (pd.DataFrame): Input dataframe
        param_hash (str): Hash string for caching
        config (dict): Configuration dictionary
        output_dir (str): Base directory to store outputs
        use_mlflow (bool): Whether to log artifacts to MLflow

    Returns:
        tuple: Transformed dataframe and path to output folder
    """
    step_dir = os.path.join(output_dir, f"step1_{param_hash}")
    final_csv = os.path.join(step_dir, f"step1_feature_engineered_{param_hash}.csv")
    registry_file = os.path.join(step_dir, f"registry_{param_hash}.json")
    meta_file = os.path.join(step_dir, f"feature_metadata_{param_hash}.json")
    manifest_file = os.path.join(step_dir, "manifest.json")

    if os.path.exists(manifest_file) and os.path.exists(final_csv):
        print(f"[STEP 1] Skipping: cached result at {step_dir}")
        return pd.read_csv(final_csv), step_dir

    os.makedirs(step_dir, exist_ok=True)

    # Add derived features
    df, derived = add_derived_features(df)

    # Drop low-variance features
    df, dropped = drop_low_variance_features(df)

    registry = {
        "param_hash": param_hash,
        "final_features": list(df.columns),
        "derived_features": derived,
        "dropped_features": dropped
    }
    with open(registry_file, "w") as f:
        json.dump(registry, f, indent=2)

    metadata = {
        "param_hash": param_hash,
        "columns": {
            col: {
                "dtype": str(df[col].dtype),
                "cardinality": int(df[col].nunique()),
                "nulls": int(df[col].isnull().sum())
            } for col in df.columns
        }
    }
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    df.to_csv(final_csv, index=False)

    manifest = {
        "step": "feature_engineering",
        "param_hash": param_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "feature_engineered_csv": final_csv,
            "registry": registry_file,
            "metadata": meta_file
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    if use_mlflow:
        with mlflow.start_run(run_name=f"Step1_FeatureEng_{param_hash}"):
            mlflow.set_tags({"step": "feature_engineering", "hash": param_hash})
            mlflow.log_params({"num_final_features": len(df.columns)})
            mlflow.log_artifacts(step_dir, artifact_path="feature_engineering")

    return df, step_dir


if __name__ == "__main__":
    df = load_data()
    config = {
        "columns": sorted(list(df.columns)),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }
    param_hash = make_param_hash(config)
    df, step_dir = run_feature_engineering(df, param_hash, config, use_mlflow=True)
    print(f"Feature engineering completed. Engineered data saved to '{step_dir}' directory.")
    print(df.head())
# -*- coding: utf-8 -*-
# Step 2 - Data Partitioning
# import sqlite3
import pandas as pd
# import numpy as np
import os
import json
# import hashlib  # Add import for hashing
from datetime import datetime, timezone  # Update import to include timezone
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split  # Add this import
# from typing import Literal
from utils import make_param_hash, load_data  # Assuming this function is in utils.py
from feature_engineering import run_feature_engineering
import mlflow  # Add import for MLflow


def get_stratify_keys(df: pd.DataFrame, target_col: str, max_cardinality: int = 10) -> pd.Series:
    """
    Creates a multi-class stratification key based on target and low-cardinality categorical features.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Name of the target column.
        max_cardinality (int): Max unique values for categorical stratification features.

    Returns:
        pd.Series: Combined stratification key.
    """
    stratify_cols = [
        col for col in df.columns
        if df[col].dtype == "object" and df[col].nunique() <= max_cardinality and col != target_col
    ]
    return df[target_col].astype(str) + "_" + df[stratify_cols].astype(str).agg("_".join, axis=1) if stratify_cols else df[target_col].astype(str)


def run_partitioning(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    param_hash: str,  # Add param_hash parameter
    output_path: str = "artifacts/step2",
    seed: int = 42,
    train_size: float = 0.6,
    val_size: float = 0.1,
    test_size: float = 0.3,
    stratify_cardinality_threshold: int = 10,
    use_mlflow: bool = False  # Add use_mlflow parameter
) -> tuple[dict[str, pd.DataFrame], str]:
    """
    Partitions dataset into train, val, test sets with stratification and stores partition IDs.
    Now includes hash-driven caching and checkpointing.

    Args:
        df (pd.DataFrame): The feature-engineered dataframe.
        id_col (str): Column name to track row IDs for reproducibility.
        target_col (str): Target variable to stratify on.
        param_hash (str): Configuration hash for caching.
        output_path (str): Where to store partition metadata and splits.
        seed (int): Random seed for reproducibility.
        train_size (float): Fraction for training set.
        val_size (float): Fraction for validation set.
        test_size (float): Fraction for testing set.
        stratify_cardinality_threshold (int): Max cardinality for stratification helpers.
        use_mlflow (bool): Whether to log results to MLflow.

    Returns:
        tuple: (Dictionary containing train, val, test dataframes, artifact path).
    """
    step_dir = os.path.join(output_path, f"step2_{param_hash}")  # Use param_hash for directory naming

    # Check for cached partition
    if os.path.exists(os.path.join(step_dir, "manifest.json")):
        print(f"[STEP2] Skipping: Found cached partition in {step_dir}")
        splits = {
            "train": pd.read_csv(os.path.join(step_dir, f"train_{param_hash}.csv")),
            "validation": pd.read_csv(os.path.join(step_dir, f"validation_{param_hash}.csv")),
            "test": pd.read_csv(os.path.join(step_dir, f"test_{param_hash}.csv")),
        }
        return splits, step_dir

    os.makedirs(step_dir, exist_ok=True)

    stratify_key = get_stratify_keys(df, target_col, max_cardinality=stratify_cardinality_threshold)

    # Check if stratification is feasible
    class_counts = stratify_key.value_counts()
    if (class_counts < 2).any():
        print("[WARNING] Some classes in the stratification key have fewer than 2 members. Falling back to non-stratified splitting.")
        stratify_key = None

    # First split: train+val vs. test
    if stratify_key is not None:
        sss_1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(sss_1.split(df, stratify_key))
    else:
        train_val_idx, test_idx = train_test_split(
            df.index, test_size=test_size, random_state=seed
        )

    train_val_df = df.loc[train_val_idx].copy()  # Use .loc for index alignment
    test_df = df.loc[test_idx].copy()  # Use .loc for index alignment

    # Adjust val fraction based on remaining set
    val_frac_within_train_val = val_size / (train_size + val_size)

    stratify_key_train_val = get_stratify_keys(train_val_df, target_col, max_cardinality=stratify_cardinality_threshold) if stratify_key is not None else None
    if stratify_key_train_val is not None:
        class_counts_train_val = stratify_key_train_val.value_counts()
        if (class_counts_train_val < 2).any():
            print("[WARNING] Some classes in the train+val stratification key have fewer than 2 members. Falling back to non-stratified splitting.")
            stratify_key_train_val = None

    if stratify_key_train_val is not None:
        sss_2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_within_train_val, random_state=seed)
        train_idx, val_idx = next(sss_2.split(train_val_df, stratify_key_train_val))
    else:
        train_idx, val_idx = train_test_split(
            train_val_df.index, test_size=val_frac_within_train_val, random_state=seed
        )

    train_df = train_val_df.loc[train_idx].copy()  # Use .loc for index alignment
    val_df = train_val_df.loc[val_idx].copy()  # Use .loc for index alignment

    # Save partitioned data with hash ID in filenames
    train_df.to_csv(os.path.join(step_dir, f"train_{param_hash}.csv"), index=False)
    val_df.to_csv(os.path.join(step_dir, f"validation_{param_hash}.csv"), index=False)
    test_df.to_csv(os.path.join(step_dir, f"test_{param_hash}.csv"), index=False)

    # Save partition IDs
    with open(os.path.join(step_dir, f"partition_ids_{param_hash}.json"), "w") as f:  # Update filename with param_hash
        json.dump({
            "train": train_df[id_col].tolist(),
            "validation": val_df[id_col].tolist(),
            "test": test_df[id_col].tolist()
        }, f, indent=2)

    # Save stratify keys for audit
    strat_df = pd.DataFrame({
        id_col: df[id_col],
        "stratify_key": stratify_key
    })
    strat_df.to_csv(os.path.join(step_dir, f"stratify_keys_{param_hash}.csv"), index=False)  # Update filename with param_hash

    # Save manifest
    manifest = {
        "step": "partitioning",
        "timestamp": datetime.now(timezone.utc).isoformat(),  # Use timezone-aware UTC datetime
        "param_hash": param_hash,  # Include param_hash in manifest
        "id_col": id_col,
        "target_col": target_col,
        "seed": seed,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "stratify_cardinality_threshold": stratify_cardinality_threshold,
        "output_dir": step_dir,
        "outputs": {
            "train": f"train_{param_hash}.csv",
            "validation": f"validation_{param_hash}.csv",
            "test": f"test_{param_hash}.csv",
            "ids": f"partition_ids_{param_hash}.json",
            "stratify_keys": f"stratify_keys_{param_hash}.csv"
        }
    }
    with open(os.path.join(step_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Add MLflow logging if enabled
    if use_mlflow:
        with mlflow.start_run(run_name=f"Step2_Partitioning_{param_hash}"):  # Use param_hash in run name
            mlflow.set_tags({"step": "partitioning", "hash": param_hash})
            mlflow.log_params({
                "id_col": id_col,
                "target_col": target_col,
                "seed": seed,
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
                "stratify_cardinality_threshold": stratify_cardinality_threshold,
                "num_train": len(train_df),
                "num_val": len(val_df),
                "num_test": len(test_df)
            })
            mlflow.log_artifacts(step_dir, artifact_path="partitioning")

    return {"train": train_df, "validation": val_df, "test": test_df}, step_dir


if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("artifacts/step2", exist_ok=True)
    df = load_data()
    print(f"[DEBUG] Loaded data type: {type(df)}")  # Debugging statement
    hash = make_param_hash({
        "columns": sorted(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    })
    print(f"[DEBUG] Hash for feature engineering: {hash}")  # Debugging statement
    df_result = run_feature_engineering(df, use_mlflow=True, param_hash=hash, config={
        "columns": sorted(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    })
    print(f"[DEBUG] Feature-engineering df result data type: {type(df_result)}")  # Debugging statement
    # Check if df_result is a tuple (e.g., (df, step_dir))
    # If it is a tuple, unpack it to get the DataFrame and step_dir
    if isinstance(df_result, tuple):
        df = df_result[0]  # Assuming the first element is the DataFrame
        print(f"[DEBUG] Unpacked feature-engineered data type: {type(df)}")
    else:
        df = df_result
        print(f"[DEBUG] Feature-engineered data type: {type(df)}")

    split_dict, artifact_path = run_partitioning(
        df=df,
        id_col="transaction_id",
        target_col="is_fraud",
        param_hash=make_param_hash({
            "id_col": "transaction_id",
            "target_col": "is_fraud",
            "seed": 123,
            "train_size": 0.6,
            "val_size": 0.1,
            "test_size": 0.3,
            "stratify_cardinality_threshold": 10
        }),
        output_path="artifacts/step2",
        seed=123,
        train_size=0.6,
        val_size=0.1,
        test_size=0.3,
        stratify_cardinality_threshold=10
    )

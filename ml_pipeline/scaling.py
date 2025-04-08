import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
from .utils import make_param_hash
from .utils import log_registry


def standardize_dataframe(
    df: pd.DataFrame,
    numeric_cols: list[str],
    center_stats: pd.Series,
    scale_stats: pd.Series
) -> pd.DataFrame:
    """
    Apply centering and scaling to a dataframe using provided stats.
    Only scales the specified numeric columns, leaving others untouched.

    Args:
        df: Input DataFrame to scale.
        numeric_cols: List of numeric column names to be scaled.
        center_stats: Series of center values (mean or median).
        scale_stats: Series of scale values (std or IQR).

    Returns:
        Scaled DataFrame with specified numeric columns standardized.
    """
    df_scaled = df.copy()
    # Only apply scaling to specified numeric columns that exist in the dataframe
    valid_numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if valid_numeric_cols:
        df_scaled[valid_numeric_cols] = (df[valid_numeric_cols] - center_stats[valid_numeric_cols]) / scale_stats[valid_numeric_cols]
    return df_scaled


def scaling(self) -> None:
    """
    Step 4: Apply centering and scaling to numeric features using train-only stats.
    Centering: mean or median (T1)
    Scaling: standard deviation or IQR (S1)
    The target column is preserved without any scaling.

    Updates:
        self.dataframes["train_sca"], ["val_sca"], ["test_sca"]
        self.paths["scaling"], self.hashes["scaling"]
    """
    step = "scaling"
    
    # Use consistent naming from numeric_conversion step
    train_df = self.dataframes["train_num"]
    val_df = self.dataframes["val_num"]
    test_df = self.dataframes["test_num"]
    excluded_df = self.dataframes.get("excluded_num", None)

    # Print available keys for debugging
    print(f"[{step.upper()}] Available dataframe keys: {list(self.dataframes.keys())}")
    
    target_col = self.config["target_col"]
    id_col = self.config["id_col"]
    
    # Store target and ID values separately before scaling
    train_target = train_df[target_col] if target_col in train_df.columns else None
    val_target = val_df[target_col] if target_col in val_df.columns else None
    test_target = test_df[target_col] if target_col in test_df.columns else None
    excluded_target = excluded_df[target_col] if excluded_df is not None and target_col in excluded_df.columns else None
    
    train_id = train_df[id_col] if id_col in train_df.columns else None
    val_id = val_df[id_col] if id_col in val_df.columns else None
    test_id = test_df[id_col] if id_col in test_df.columns else None
    excluded_id = excluded_df[id_col] if excluded_df is not None and id_col in excluded_df.columns else None
    
    # Exclude target and ID columns from scaling
    exclude_cols = [col for col in [target_col, id_col] if col in train_df.columns]  # + [col for col in [target_col, id_col] if col in excluded_df.columns] if excluded_df is not None else []

    config = {
        "s1": self.config["s1"],  # True → std, False → IQR
        "t1": self.config["t1"]   # True → mean, False → median
    }
    param_hash = make_param_hash(config)
    step_dir = os.path.join("artifacts", f"{step}_{param_hash}")
    manifest_file = os.path.join(step_dir, "manifest.json")

    if os.path.exists(manifest_file):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = param_hash
        for split in ["train_sca", "val_sca", "test_sca", "excluded_sca"]:
            if split in self.dataframes:
                continue
            split_file = split.replace("_sca", "")  # Get base name for file
            self.dataframes[split] = pd.read_csv(os.path.join(step_dir, f"{split_file}_scaled_{param_hash}.csv"))
        return

    os.makedirs(step_dir, exist_ok=True)

    # By this point, all columns should be numeric, no need for selection
    # But we'll keep the check for safety and log a warning if non-numeric columns exist
    non_numeric_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"[WARNING] Found non-numeric columns after numeric conversion: {non_numeric_cols}")
        print("These columns will be excluded from scaling.")
    
    numeric_cols = [col for col in train_df.select_dtypes(include=[np.number]).columns.tolist() 
                   if col not in exclude_cols]

    # Compute translation and scale stats from train only
    center_func = np.mean if config["t1"] else np.median
    scale_func = np.std if config["s1"] else lambda x: np.subtract(*np.percentile(x, [75, 25]))

    center_stats = train_df[numeric_cols].agg(center_func)
    scale_stats = train_df[numeric_cols].agg(scale_func)
    scale_stats.replace(0, 1.0, inplace=True)  # avoid division by zero

    df_train_scaled = standardize_dataframe(train_df, numeric_cols, center_stats, scale_stats)
    df_val_scaled = standardize_dataframe(val_df, numeric_cols, center_stats, scale_stats)
    df_test_scaled = standardize_dataframe(test_df, numeric_cols, center_stats, scale_stats)
    df_excluded_scaled = standardize_dataframe(excluded_df, numeric_cols, center_stats, scale_stats) if excluded_df is not None else None
    # Add back the target and ID columns
    if train_target is not None:
        df_train_scaled[target_col] = train_target
    if val_target is not None:
        df_val_scaled[target_col] = val_target
    if test_target is not None:
        df_test_scaled[target_col] = test_target
    if excluded_target is not None and df_excluded_scaled is not None:
        df_excluded_scaled[target_col] = excluded_target
        
    if train_id is not None:
        df_train_scaled[id_col] = train_id
    if val_id is not None:
        df_val_scaled[id_col] = val_id
    if test_id is not None:
        df_test_scaled[id_col] = test_id
    if excluded_id is not None and df_excluded_scaled is not None:
        df_excluded_scaled[id_col] = excluded_id

    # Save to CSVs with consistent naming
    df_train_scaled.to_csv(os.path.join(step_dir, f"train_scaled_{param_hash}.csv"), index=False)
    df_val_scaled.to_csv(os.path.join(step_dir, f"val_scaled_{param_hash}.csv"), index=False)
    df_test_scaled.to_csv(os.path.join(step_dir, f"test_scaled_{param_hash}.csv"), index=False)
    # if df_excluded_scaled is not None:
    df_excluded_scaled.to_csv(os.path.join(step_dir, f"excluded_scaled_{param_hash}.csv"), index=False) if df_excluded_scaled is not None else None

    # Save transformation stats
    stats_file = os.path.join(step_dir, f"scaling_stats_{param_hash}.json")
    stats = {
        "center_function": "mean" if config["t1"] else "median",
        "scale_function": "std" if config["s1"] else "iqr",
        "center": center_stats.to_dict(),
        "scale": scale_stats.to_dict()
    }
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "train_scaled_csv": f"train_scaled_{param_hash}.csv",
            "val_scaled_csv": f"val_scaled_{param_hash}.csv",
            "test_scaled_csv": f"test_scaled_{param_hash}.csv",
            "excluded_scaled_csv": f"excluded_scaled_{param_hash}.csv",
            "scaling_stats_json": f"scaling_stats_{param_hash}.json"
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    log_registry(step, param_hash, config, step_dir)

    self.dataframes["train_sca"] = df_train_scaled
    self.dataframes["val_sca"] = df_val_scaled
    self.dataframes["test_sca"] = df_test_scaled
    self.dataframes["excluded_sca"] = df_excluded_scaled  if df_excluded_scaled is not None else None
    self.artifacts[step] = {
        "train_scaled_csv": os.path.join(step_dir, f"train_scaled_{param_hash}.csv"),
        "val_scaled_csv": os.path.join(step_dir, f"val_scaled_{param_hash}.csv"),
        "test_scaled_csv": os.path.join(step_dir, f"test_scaled_{param_hash}.csv"),
        "excluded_scaled_csv": os.path.join(step_dir, f"excluded_scaled_{param_hash}.csv"),
        "scaling_stats_json": os.path.join(step_dir, f"scaling_stats_{param_hash}.json")
    }
    self.paths[step] = step_dir
    self.hashes[step] = param_hash

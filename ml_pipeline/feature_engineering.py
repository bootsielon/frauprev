import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import mlflow
from .utils import make_param_hash
from .utils import log_registry


def feature_engineering(self) -> None:
    """
    Step 1: Add derived features, drop zero-variance or constant columns.
    Updates: self.dataframes["feature_engineered"], self.paths["feature_engineering"], self.hashes["feature_engineering"]
    """
    df = self.dataframes["raw"]
    step = "feature_engineering"

    config = {
        "columns": sorted(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }
    param_hash = make_param_hash(config)
    step_dir = os.path.join("artifacts", f"{step}_{param_hash}")
    manifest_file = os.path.join(step_dir, "manifest.json")

    if os.path.exists(manifest_file):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = param_hash
        self.dataframes["feature_engineered"] = pd.read_csv(os.path.join(step_dir, f"{step}_{param_hash}.csv"))
        return

    os.makedirs(step_dir, exist_ok=True)
    df_fe = df.copy()

    # Feature engineering example: timestamp-based
    if "timestamp" in df_fe.columns:
        df_fe["transaction_hour"] = pd.to_datetime(df_fe["timestamp"]).dt.hour
        df_fe["transaction_dayofweek"] = pd.to_datetime(df_fe["timestamp"]).dt.dayofweek

    if "account_creation_date_client" in df_fe.columns:
        df_fe["client_account_age_days"] = (
            pd.to_datetime(df_fe["timestamp"]) - pd.to_datetime(df_fe["account_creation_date_client"])
        ).dt.days

    if "account_creation_date_merchant" in df_fe.columns:
        df_fe["merchant_account_age_days"] = (
            pd.to_datetime(df_fe["timestamp"]) - pd.to_datetime(df_fe["account_creation_date_merchant"])
        ).dt.days

    # Drop constant columns (0 variance or same value)
    dropped = []
    for col in df_fe.columns:
        if df_fe[col].nunique(dropna=False) <= 1:
            dropped.append(col)
    df_fe.drop(columns=dropped, inplace=True)

    output_csv = os.path.join(step_dir, f"{step}_{param_hash}.csv")
    drop_log = os.path.join(step_dir, f"dropped_features_{param_hash}.json")

    df_fe.to_csv(output_csv, index=False)
    with open(drop_log, "w") as f:
        json.dump({"dropped_features": dropped}, f, indent=2)

    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "engineered_csv": output_csv,
            "dropped_features": drop_log
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    log_registry(step, param_hash, config, step_dir)
    self.dataframes["feature_engineered"] = df_fe
    self.paths[step] = step_dir
    self.hashes[step] = param_hash

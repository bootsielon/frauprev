# === ml_pipeline/feature_engineering.py ===============================
"""
Feature‑engineering step.

Conforms to the pipeline spec:
• All artefacts live in `artifacts/run_<global_hash>/feature_engineering/`
  (no per‑step hashes).
• Training creates artefacts; inference reuses or derives from training
  artefacts exactly as described in spec §5.
• Legacy objects (`self.hashes`, `make_param_hash`, etc.) kept but
  commented‑out for reference.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import mlflow
import pandas as pd

# absolute import, per additional rule
from ml_pipeline.utils import (
    convert_numpy_types,
    log_registry, make_param_hash
) 


def feature_engineering(self) -> None:  # noqa: C901
    """
    Step 1: create derived features & drop zero‑variance columns.

    Updates:
        self.dataframes["feature_engineered"]
        self.paths["feature_engineering"]
        self.artifacts["feature_engineering"]
    """
    step              = "feature_engineering"
    train_mode        = self.train_mode
    df                = self.dataframes["raw"]
    run_dir           = self.run_dir                      # current (train or inference) run
    step_dir          = os.path.join(run_dir, step)       # always exist for this run
    train_step_dir    = os.path.join(                     # where training artefacts live
        "artifacts", f"run_{self.global_train_hash}", step
    )

    os.makedirs(step_dir, exist_ok=True)

    manifest_file         = os.path.join(step_dir, "manifest.json")
    train_manifest_file   = os.path.join(train_step_dir, "manifest.json")

    # ────────────────────────────────────────────────────────────────
    # SHORT‑CIRCUIT IF ARTEFACTS ALREADY EXIST (rule 5‑1)
    # ────────────────────────────────────────────────────────────────
    if os.path.exists(manifest_file):
        print(f"[{step.upper()}] Skipping — artefacts already present at {step_dir}")
        with open(manifest_file) as fh:
            manifest = json.load(f)
        if train_mode:
            self.dataframes["feature_engineered"] = pd.read_csv(
                manifest["outputs"]["engineered_csv"]
            )
        else:
            self.dataframes["feature_engineered"] = pd.read_csv(
                manifest["outputs"]["engineered_csv"]
            )
        self.paths[step]     = step_dir
        self.artifacts[step] = manifest["outputs"]
        return

    # ────────────────────────────────────────────────────────────────
    # INFERENCE: NEED TRAIN ARTEFACTS
    # ────────────────────────────────────────────────────────────────
    if not train_mode:
        if not os.path.exists(train_manifest_file):
            raise FileNotFoundError(
                f"[{step}] No training artefacts found at {train_step_dir}"
            )
        with open(train_manifest_file) as f:
            train_manifest = json.load(f)
        dropped = train_manifest["outputs"]["dropped_features"]
    else:
        dropped: list[str] = []  # will be filled below

    # ────────────────────────────────────────────────────────────────
    # FEATURE ENGINEERING
    # ────────────────────────────────────────────────────────────────
    df_fe = df.copy()

    # Example timestamp features
    if "timestamp" in df_fe.columns:
        ts = pd.to_datetime(df_fe["timestamp"])
        df_fe["transaction_hour"]      = ts.dt.hour
        df_fe["transaction_dayofweek"] = ts.dt.dayofweek

    # Account‑age derived features
    for col in ["account_creation_date_client", "account_creation_date_merchant"]:
        if col in df_fe.columns:
            dt = pd.to_datetime(df_fe[col], errors="coerce")
            df_fe[f"{col}_year"]       = dt.dt.year
            df_fe[f"{col}_month"]      = dt.dt.month
            df_fe[f"{col}_day"]        = dt.dt.day
            df_fe[f"{col}_hour"]       = dt.dt.hour
            df_fe[f"{col}_dayofweek"]  = dt.dt.dayofweek
            if "timestamp" in df_fe.columns:
                df_fe[f"{col}_age_days"]  = (pd.to_datetime(df_fe["timestamp"]) - dt).dt.days
                df_fe[f"{col}_age_years"] = df_fe[f"{col}_age_days"] / 365.25

    # Drop constant / zero‑variance columns
    if train_mode:
        for col in df_fe.columns:
            if df_fe[col].nunique(dropna=False) <= 1:
                dropped.append(col)
        df_fe.drop(columns=dropped, inplace=True)
    else:
        # ensure exactly the columns dropped during training are removed
        df_fe.drop(columns=[c for c in dropped if c in df_fe.columns], inplace=True)

    # ────────────────────────────────────────────────────────────────
    # SAVE OUTPUTS
    # ────────────────────────────────────────────────────────────────
    engineered_csv = os.path.join(step_dir, "engineered.csv")
    df_fe.to_csv(engineered_csv, index=False)

    drop_log = os.path.join(step_dir, "dropped_features.json")
    with open(drop_log, "w") as fh:
        json.dump({"dropped_features": dropped}, fh, indent=2)

    # Manifest
    config_summary: dict[str, Any] = {
        "columns"          : sorted(df.columns),
        "train_mode"       : train_mode,
        "target_col"       : self.config.get("target_col"),
    }
    manifest = {
        "step"        : step,
        "global_hash" : self.global_hash,
        "timestamp"   : datetime.utcnow().isoformat(),
        "config"      : convert_numpy_types(config_summary),
        "output_dir"  : step_dir,
        "outputs"     : {
            "engineered_csv"        : engineered_csv,
            "dropped_features"      : dropped,
            "dropped_features_log"  : drop_log,
            "retained_features"     : sorted(df_fe.columns),
            "initial_shape"         : df.shape,
            "final_shape"           : df_fe.shape,
        },
    }
    with open(manifest_file, "w") as fh:
        json.dump(convert_numpy_types(manifest), fh, indent=2)

    # ────────────────────────────────────────────────────────────────
    # OPTIONAL: MLflow LOGGING
    # ────────────────────────────────────────────────────────────────
    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{self.global_hash}"):
            mlflow.set_tags({"step": step, "global_hash": self.global_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    # Registry & pipeline state
    log_registry(step, self.global_hash, config_summary, step_dir)
    self.dataframes["feature_engineered"] = df_fe
    self.paths[step]                      = step_dir
    # removed: self.hashes no longer used


# ──────────────────────────────────────────────────────────────────────
# SMOKE‑TEST (spec §8)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick smoke‑test using a tiny in‑memory DataFrame.
    """
    from datetime import timedelta
    from ml_pipeline.base import MLPipeline

    now = datetime.now()
    mock = pd.DataFrame(
        {
            "client_id": [1, 2],
            "merchant_id": [10, 11],
            "amount": [100.0, 50.0],
            "is_fraud": [0, 1],
            "timestamp": [(now - timedelta(days=1)).isoformat()] * 2,
            "account_creation_date_client": [
                (now - timedelta(days=100)).isoformat()
            ] * 2,
        }
    )

    cfg = {"target_col": "is_fraud", "use_mlflow": False}

    pipe = MLPipeline(cfg, data_source="raw", raw_data=mock)
    pipe.dataframes["raw"] = mock
    pipe.feature_engineering()

    print("Engineered DF shape:", pipe.dataframes["feature_engineered"].shape)
    print("Artefacts stored at :", pipe.paths["feature_engineering"])
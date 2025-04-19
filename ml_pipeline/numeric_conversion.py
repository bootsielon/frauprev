# === ml_pipeline/numeric_conversion.py ================================
"""
Step 3 – Convert non‑numeric variables to numeric representations
(one‑hot encoding, rare‑category grouping, ID‑like dropping, imputation).

Spec compliance highlights
• Artefacts live in   artifacts/run_<global_hash>/numeric_conversion/
  (no step‑specific hashes in filenames).
• Training creates artefacts; inference *only* reuses the training ones.
• No `self.hashes` updates (legacy lines commented).
• Absolute imports & full docstrings/type hints retained/added.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import mlflow
import numpy as np
import pandas as pd

# absolute imports per spec §11
from ml_pipeline.utils import convert_numpy_types, log_registry, make_param_hash


# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE STEP
# ──────────────────────────────────────────────────────────────────────
def numeric_conversion(self) -> None:  # noqa: C901
    """
    Perform numeric conversion for train / val / test (+optional excluded)
    datasets that were produced by the *partitioning* step.

    Updates
    -------
    self.dataframes  : adds train_num / val_num / test_num / excluded_num
    self.paths       : path to artefact directory for this step
    self.artifacts   : dict of filenames written
    """
    step = "numeric_conversion"
    run_dir      = self.run_dir
    step_dir     = os.path.join(run_dir, step)
    train_dir    = os.path.join("artifacts", f"run_{self.global_train_hash}", step)

    os.makedirs(step_dir, exist_ok=True)
    manifest_fp = os.path.join(step_dir, "manifest.json")

    # --- PATCH: replace the current fast‑exit block in numeric_conversion.py ----
    # ── 1. Fast‑exit if artefacts already exist for *this* run ─────────
    if os.path.exists(manifest_fp):
        with open(manifest_fp) as fh:
            manifest = json.load(fh)

        for name, rel in manifest["outputs"].items():
            if rel is None or not name.endswith("_csv"):
                continue
            full_path = os.path.join(step_dir, rel) if not os.path.isabs(rel) else rel
            key = name.replace("_csv", "")
            self.dataframes[key] = pd.read_csv(full_path)

        self.paths[step]     = step_dir
        self.artifacts[step] = manifest["outputs"]
        print(f"[{step.upper()}] Skipping — artefacts already present at {step_dir}")
        return
    
    # ---------------------------------------------------------------------------
    # ── 2. Inference mode – just copy artefacts from training run ─────
    if not self.train_mode:
        train_manifest_fp = os.path.join(train_dir, "manifest.json")
        if not os.path.exists(train_manifest_fp):
            raise FileNotFoundError(f"[{step}] Training artefacts missing at {train_dir}")

        # copy necessary files locally
        for fname in ["train_num.csv", "val_num.csv", "test_num.csv",
                      "excluded_num.csv", "feature_names.json"]:
            src = os.path.join(train_dir, fname)
            if os.path.exists(src):
                import shutil
                shutil.copy(src, step_dir)

        # replicate manifest with new output_dir
        train_manifest = json.load(open(train_manifest_fp))
        train_manifest["output_dir"] = step_dir
        json.dump(train_manifest, open(manifest_fp, "w"), indent=2)

        # load test set (only thing inference needs)
        self.dataframes["test_num"] = pd.read_csv(os.path.join(step_dir, "test_num.csv"))
        self.paths[step]     = step_dir
        self.artifacts[step] = train_manifest["outputs"]
        print(f"[{step.upper()}] Inference: artefacts copied from training run")
        return

    # ── 3. Training mode – compute numeric conversion ─────────────────
    cfg = self.config  # shorthand
    train_df    = self.dataframes["train"]
    val_df      = self.dataframes["val"]
    test_df     = self.dataframes["test"]
    excluded_df = self.dataframes.get("excluded")

    dropped: List[str]               = []
    grouping_map: Dict               = {}
    id_like_columns: List[str]       = []
    encoded_mapping: Dict[str, Dict] = {}
    inverse_mapping: Dict[str, Dict] = {}

    # --------‑‑‑ Identify & drop constant columns ---------------------
    constant_cols: List[str] = [
        c for c in train_df.columns
        if (train_df[c].isna().all()) or (train_df[c].nunique(dropna=False) <= 1)
    ]
    dropped += constant_cols
    train_df = train_df.drop(columns=constant_cols)
    val_df   = val_df.drop(columns=[c for c in constant_cols if c in val_df])
    test_df  = test_df.drop(columns=[c for c in constant_cols if c in test_df])
    if excluded_df is not None:
        excluded_df = excluded_df.drop(columns=[c for c in constant_cols if c in excluded_df])

    # --------‑‑‑ Cardinality‑based handling of categoricals -----------
    dataset_size = len(train_df)
    c1, c2, b1, c3 = cfg["c1"], cfg["c2"], cfg["b1"], cfg["c3"]

    for col in list(train_df.columns):
        if pd.api.types.is_numeric_dtype(train_df[col]) or col == cfg["target_col"]:
            continue

        cardinality = train_df[col].nunique(dropna=False)
        col_fraction = cardinality / dataset_size

        if cardinality <= c1:
            continue  # low cardinality – leave for one‑hot
        elif col_fraction <= c2 or b1:
            # mid/high treated as mid
            top = train_df[col].value_counts().nlargest(c1).index
            train_df[col] = train_df[col].where(train_df[col].isin(top), other="Other")
            grouping_map[col] = {"strategy": "top_c1+other", "top_categories": top.tolist()}
        else:
            # high cardinality – maybe ID‑like
            log_ratio = np.log10(dataset_size) / np.log10(max(cardinality, 2))
            if cfg["id_like_exempt"] and 1 <= log_ratio <= c3:
                id_like_columns.append(col)
            dropped.append(col)

    # drop high‑card / id‑like cols
    if dropped:
        train_df = train_df.drop(columns=dropped)
        val_df   = val_df.drop(columns=[c for c in dropped if c in val_df])
        test_df  = test_df.drop(columns=[c for c in dropped if c in test_df])
        if excluded_df is not None:
            excluded_df = excluded_df.drop(columns=[c for c in dropped if c in excluded_df])

    # --------‑‑‑ Imputation ------------------------------------------
    datasets = {"train": train_df, "val": val_df, "test": test_df}
    if excluded_df is not None:
        datasets["excluded"] = excluded_df

    imputation_stats: Dict[str, Dict] = {}
    central_tendency = cfg.get("central_tendency", "median")

    # numeric
    for col in train_df.select_dtypes(include="number"):
        value = train_df[col].mean() if central_tendency == "mean" else train_df[col].median()
        for df in datasets.values():
            if col in df:
                df[col] = df[col].fillna(value)
        imputation_stats[col] = {"strategy": central_tendency, "value": value}

    # categorical
    for col in train_df.select_dtypes(exclude="number"):
        mode_val = train_df[col].mode()[0] if not train_df[col].mode().empty else "MISSING"
        for df in datasets.values():
            if col in df:
                df[f"{col}_is_NA"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(mode_val).replace("", mode_val)
        imputation_stats[col] = {"strategy": "mode", "value": mode_val}

    # --------‑‑‑ One‑hot encoding ------------------------------------
    cat_cols = train_df.select_dtypes(exclude="number").columns.tolist()
    encoded = {name: pd.get_dummies(df, columns=cat_cols, drop_first=False)
               for name, df in datasets.items()}

    # align columns across sets
    base_cols = encoded["train"].columns
    for name, df in encoded.items():
        if name == "train":
            continue
        for col in base_cols:
            if col not in df:
                df[col] = 0
        extra = [c for c in df.columns if c not in base_cols]
        if extra:
            df.drop(columns=extra, inplace=True)
        encoded[name] = df[base_cols]

    # --------‑‑‑ Save artefacts --------------------------------------
    train_encoded = encoded["train"]
    val_encoded   = encoded["val"]
    test_encoded  = encoded["test"]
    excluded_encoded = encoded.get("excluded")

    train_encoded.to_csv(os.path.join(step_dir, "train_num.csv"), index=False)
    val_encoded.to_csv(os.path.join(step_dir, "val_num.csv"), index=False)
    test_encoded.to_csv(os.path.join(step_dir, "test_num.csv"), index=False)
    if excluded_encoded is not None:
        excluded_encoded.to_csv(os.path.join(step_dir, "excluded_num.csv"), index=False)

    # manifest
    manifest = {
        "step": step,
        "global_hash": self.global_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": convert_numpy_types(cfg),
        "output_dir": step_dir,
        "outputs": {
            "train_num_csv": "train_num.csv",
            "val_num_csv": "val_num.csv",
            "test_num_csv": "test_num.csv",
            "excluded_num_csv": "excluded_num.csv" if excluded_encoded is not None else None,
            "feature_names_json": "feature_names.json",
        },
    }
    json.dump(manifest, open(manifest_fp, "w"), indent=2)

    # save feature names
    feature_names_fp = os.path.join(step_dir, "feature_names.json")
    json.dump({"feature_names": train_encoded.columns.tolist()}, open(feature_names_fp, "w"), indent=2)

    # optional MLflow
    if cfg.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{self.global_hash}"):
            mlflow.set_tags({"step": step, "global_hash": self.global_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    # registry + state
    log_registry(step, self.global_hash, cfg, step_dir)

    self.dataframes["train_num"] = train_encoded
    self.dataframes["val_num"]   = val_encoded
    self.dataframes["test_num"]  = test_encoded
    if excluded_encoded is not None:
        self.dataframes["excluded_num"] = excluded_encoded

    self.paths[step]     = step_dir
    self.artifacts[step] = manifest["outputs"]
    # removed: self.hashes no longer used


# ──────────────────────────────────────────────────────────────────────
# SMOKE‑TEST (spec §8)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Tiny smoke‑test on synthetic data.
    """
    from ml_pipeline.base import MLPipeline
    import numpy as np

    np.random.seed(0)
    df = pd.DataFrame(
        {
            "id": range(200),
            "amount": np.random.randn(200),
            "low_card": np.random.choice(list("ABC"), 200),
            "mid_card": np.random.choice([f"cat{i}" for i in range(10)], 200),
            "target": np.random.choice([0, 1], 200, p=[0.8, 0.2]),
        }
    )

    cfg = {
        "target_col": "target",
        "id_col": "id",
        "c1": 5,
        "c2": 0.2,
        "b1": True,
        "c3": 1.5,
        "id_like_exempt": True,
        "central_tendency": "median",
        "seed": 42,
        "use_mlflow": False,
    }

    pipe = MLPipeline(cfg, data_source="raw", raw_data=df)
    pipe.dataframes.update({"train": df, "val": df.sample(40), "test": df.sample(40)})
    pipe.numeric_conversion()

    print("Numeric columns:", pipe.dataframes["train_num"].shape[1])
    print("Artefacts →", pipe.paths["numeric_conversion"])
# === ml_pipeline/partitioning.py ======================================
"""
Partition the feature‑engineered dataset into train / val / test splits and
(optionally) perform class‑imbalance down‑sampling.

Key points per pipeline spec:
• Artefacts live in   artifacts/run_<global_hash>/partitioning/
  (no hashes in filenames).
• Training mode creates all outputs; inference mode *never* recomputes
  splits—it reuses artefacts from the referenced training run.
• No `self.hashes`; legacy updates are commented‑out.
• All filenames are deterministic and hash‑free.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

# absolute imports (spec §11)
from ml_pipeline.utils import log_registry, make_param_hash  # make_param_hash only for manifest

# ──────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS  (signatures unchanged – internals updated)
# ──────────────────────────────────────────────────────────────────────
def identify_stratification_columns(
    df: pd.DataFrame,
    target: str,
    use_stratification: bool,
    cardinality_threshold: int,
) -> List[str]:
    """Return columns to use for stratification (target always included)."""
    stratify_cols = [target]
    if use_stratification:
        stratify_cols.extend(
            [c for c in df.columns if c != target and df[c].nunique() <= cardinality_threshold]
        )
    return stratify_cols


def identify_classes(
    df: pd.DataFrame, target: str, step: str
) -> Tuple[int, int, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Detect minority / majority class and return helper dataframes."""
    class_counts = df[target].value_counts()
    if len(class_counts) != 2:
        print(f"[{step.upper()}] Warning: expected binary classification, found {len(class_counts)} classes")

    minority_class = class_counts.index[-1]  # least frequent
    majority_class = class_counts.index[0]   # most frequent

    return (
        minority_class,
        majority_class,
        df[df[target] == minority_class],
        df[df[target] == majority_class],
        class_counts,
    )


# --- PATCH for ml_pipeline/partitioning.py ----------------------------
def perform_stratified_downsampling(
    df_majority: pd.DataFrame,
    df_minority: pd.DataFrame,
    strat_cols_for_downsampling: List[str],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Down‑sample majority class so its categorical‑strata distribution matches
    the minority class.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (downsampled_majority , excluded_majority)
    """
    majority_key  = df_majority[strat_cols_for_downsampling].astype(str).agg("_".join, axis=1)
    minority_dist = (
        df_minority[strat_cols_for_downsampling]
        .astype(str)
        .agg("_".join, axis=1)
        .value_counts(normalize=True)
    )

    collected_down  : list[pd.DataFrame] = []
    collected_excl : list[pd.DataFrame] = []

    for stratum, prop in minority_dist.items():
        stratum_df = df_majority[majority_key == stratum]
        if stratum_df.empty:
            continue

        n_samples = max(1, int(prop * len(df_minority)))
        if len(stratum_df) > n_samples:
            sampled = stratum_df.sample(n=n_samples, random_state=seed)
            collected_down.append(sampled)
            collected_excl.append(stratum_df.drop(sampled.index))
        else:
            collected_down.append(stratum_df)

    df_majority_downsampled = (
        pd.concat(collected_down, ignore_index=True) if collected_down else df_majority.iloc[0:0]
    )
    df_excluded = (
        pd.concat(collected_excl, ignore_index=True) if collected_excl else df_majority.iloc[0:0]
    )

    return df_majority_downsampled, df_excluded
# ----------------------------------------------------------------------

def balance_samples(
    df_majority_downsampled: pd.DataFrame,
    df_minority: pd.DataFrame,
    df_majority: pd.DataFrame,
    df_excluded: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure majority = minority sample count after down‑sampling."""
    if len(df_majority_downsampled) > len(df_minority):
        excess_idx               = df_majority_downsampled.sample(
            n=len(df_majority_downsampled) - len(df_minority), random_state=seed
        ).index
        df_excluded              = pd.concat([df_excluded, df_majority_downsampled.loc[excess_idx]])
        df_majority_downsampled  = df_majority_downsampled.drop(excess_idx)
    elif len(df_majority_downsampled) < len(df_minority):
        shortage = len(df_minority) - len(df_majority_downsampled)
        if len(df_excluded) >= shortage:
            addl            = df_excluded.sample(n=shortage, random_state=seed)
            df_majority_downsampled = pd.concat([df_majority_downsampled, addl])
            df_excluded     = df_excluded.drop(addl.index)
        else:
            addl_needed     = shortage - len(df_excluded)
            df_majority_downsampled = pd.concat([df_majority_downsampled, df_excluded])
            df_excluded     = pd.DataFrame(columns=df_majority.columns)
            addl            = df_majority.sample(n=addl_needed, random_state=seed)
            df_majority_downsampled = pd.concat([df_majority_downsampled, addl])
            df_excluded     = df_majority.drop(df_majority_downsampled.index)
    return df_majority_downsampled, df_excluded


def random_downsample(
    df_majority: pd.DataFrame, df_minority: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Random down‑sample majority class to minority size."""
    sampled = df_majority.sample(n=len(df_minority), random_state=seed)
    return sampled, df_majority.drop(sampled.index)


def perform_data_splits(
    df: pd.DataFrame,
    stratify_cols: List[str],
    test_size: float,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train / val / test DataFrames with stratification."""
    strat_key = df[stratify_cols].astype(str).agg("_".join, axis=1)

    df_train_val, df_test = train_test_split(
        df, test_size=test_size, stratify=strat_key, random_state=seed
    )

    strat_key_tv = df_train_val[stratify_cols].astype(str).agg("_".join, axis=1)
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_ratio, stratify=strat_key_tv, random_state=seed
    )

    return df_train, df_val, df_test


def save_outputs(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    df_excluded: pd.DataFrame,
    id_col: str,
    step_dir: str,
) -> None:
    """
    Persist CSVs for all splits and an ID map (hash‑free filenames).
    """
    df_train.to_csv(os.path.join(step_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(step_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(step_dir, "test.csv"), index=False)
    df_excluded.to_csv(os.path.join(step_dir, "excluded_majority.csv"), index=False)

    id_map: Dict[str, List] = {
        "train": df_train[id_col].tolist(),
        "val": df_val[id_col].tolist(),
        "test": df_test[id_col].tolist(),
        "excluded_majority": df_excluded[id_col].tolist(),
    }
    with open(os.path.join(step_dir, "id_partition_map.json"), "w") as fh:
        json.dump(id_map, fh, indent=2)


def create_manifest(step: str, config: dict, step_dir: str) -> None:
    """Write a manifest.json describing outputs and settings."""
    manifest = {
        "step": step,
        "global_hash": config["global_hash"],
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "train_csv": "train.csv",
            "val_csv": "val.csv",
            "test_csv": "test.csv",
            "excluded_majority_csv": "excluded_majority.csv",
            "id_partition_map_json": "id_partition_map.json",
        },
    }
    with open(os.path.join(step_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)


def load_checkpoint(step_dir: str) -> Dict[str, pd.DataFrame]:
    """Load previously‑saved CSVs if they exist (hash‑free filenames)."""
    frames: Dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test", "excluded_majority"]:
        fp = os.path.join(step_dir, f"{split}.csv")
        if os.path.exists(fp):
            frames[split if split != "excluded_majority" else "excluded"] = pd.read_csv(fp)
    return frames


# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE STEP
# ──────────────────────────────────────────────────────────────────────
def partitioning(self) -> None:  # noqa: C901
    """
    Partition feature‑engineered data into train/val/test sets adhering to the
    pipeline spec (see module docstring).
    """
    step = "partitioning"
    run_dir = self.run_dir
    step_dir = os.path.join(run_dir, step)
    train_step_dir = os.path.join("artifacts", f"run_{self.global_train_hash}", step)

    os.makedirs(step_dir, exist_ok=True)
    manifest_path = os.path.join(step_dir, "manifest.json")

    # ── 1. Fast‑exit if artefacts already exist for this run ──────────
    if os.path.exists(manifest_path):
        print(f"[{step.upper()}] Skipping — artefacts already present at {step_dir}")
        self.paths[step] = step_dir
        self.dataframes.update(load_checkpoint(step_dir))
        self.artifacts[step] = json.load(open(manifest_path))["outputs"]
        return

    # ── 2. Inference mode: copy artefacts from training run ───────────
    if not self.train_mode:
        train_manifest_path = os.path.join(train_step_dir, "manifest.json")
        if not os.path.exists(train_manifest_path):
            raise FileNotFoundError(f"[{step}] Training artefacts missing at {train_step_dir}")
        # Copy CSVs locally to keep run self‑contained
        for fname in ["train.csv", "val.csv", "test.csv", "excluded_majority.csv", "id_partition_map.json"]:
            src = os.path.join(train_step_dir, fname)
            if os.path.exists(src):
                dst = os.path.join(step_dir, fname)
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy(src, dst)
        # Re‑emit manifest pointing to the new run dir
        train_manifest = json.load(open(train_manifest_path))
        train_manifest["output_dir"] = step_dir
        json.dump(train_manifest, open(manifest_path, "w"), indent=2)

        self.dataframes.update(load_checkpoint(step_dir))
        self.paths[step] = step_dir
        self.artifacts[step] = train_manifest["outputs"]
        print(f"[{step.upper()}] Inference: artefacts copied from training run")
        return

    # ── 3. Training mode: compute splits ─────────────────────────────
    df                 = self.dataframes["feature_engineered"]
    target             = self.config["target_col"]
    id_col             = self.config["id_col"]
    seed               = self.config["seed"]
    use_strat          = self.config["use_stratification"]
    use_downsampling   = self.config.get("use_downsampling", True)
    card_thresh        = self.config["stratify_cardinality_threshold"]

    stratify_cols = identify_stratification_columns(df, target, use_strat, card_thresh)

    df_for_split = df.copy()
    df_excluded  = pd.DataFrame(columns=df.columns)

    if use_downsampling:
        (minority_class, majority_class,
         df_minority, df_majority, _) = identify_classes(df, target, step)

        if len(df_majority) >= len(df_minority):
            if use_strat and [c for c in stratify_cols if c != target]:
                strat_cols_ds = [c for c in stratify_cols if c != target]
                df_maj_ds, df_excluded = perform_stratified_downsampling(
                    df_majority, df_minority, strat_cols_ds, seed
                )
                df_maj_ds, df_excluded = balance_samples(
                    df_maj_ds, df_minority, df_majority, df_excluded, seed
                )
            else:
                df_maj_ds, df_excluded = random_downsample(df_majority, df_minority, seed)

            df_for_split = pd.concat([df_minority, df_maj_ds]).sample(frac=1, random_state=seed)

    val_ratio = self.config["val_size"] / (self.config["train_size"] + self.config["val_size"])
    df_train, df_val, df_test = perform_data_splits(
        df_for_split, stratify_cols, self.config["test_size"], val_ratio, seed
    )

    # ── 4. Save artefacts & manifest ─────────────────────────────────
    save_outputs(df_train, df_val, df_test, df_excluded, id_col, step_dir)
    create_manifest(step, self.config | {"global_hash": self.global_hash}, step_dir)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{self.global_hash}"):
            mlflow.set_tags({"step": step, "global_hash": self.global_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    # Registry + pipeline state
    log_registry(step, self.global_hash, self.config, step_dir)
    self.dataframes.update(
        {"train": df_train, "val": df_val, "test": df_test, "excluded": df_excluded}
    )
    self.paths[step] = step_dir
    # removed: self.hashes no longer used
    print(f"[{step.upper()}] Partitioning completed — artefacts at {step_dir}")


# ──────────────────────────────────────────────────────────────────────
# SMOKE‑TEST (spec §8)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run a quick end‑to‑end partitioning smoke‑test with synthetic data.
    """
    import numpy as np
    from ml_pipeline.base import MLPipeline

    np.random.seed(0)
    n = 500
    df_mock = pd.DataFrame(
        {
            "transaction_id": range(n),
            "amount": np.random.uniform(10, 1000, n),
            "is_fraud": np.random.choice([0, 1], n, p=[0.8, 0.2]),
            "client_id": np.random.randint(1, 15, n),
            "merchant_id": np.random.randint(100, 110, n),
        }
    )

    cfg = {
        "target_col": "is_fraud",
        "id_col": "transaction_id",
        "seed": 42,
        "use_stratification": True,
        "use_downsampling": True,
        "stratify_cardinality_threshold": 10,
        "train_size": 0.7,
        "val_size": 0.15,
        "test_size": 0.15,
        "use_mlflow": False,
    }

    pipe = MLPipeline(cfg, data_source="raw", raw_data=df_mock)
    pipe.dataframes["feature_engineered"] = df_mock
    pipe.partitioning()

    for split in ["train", "val", "test"]:
        print(split, pipe.dataframes[split].shape)
    print("artefacts ->", pipe.paths["partitioning"])
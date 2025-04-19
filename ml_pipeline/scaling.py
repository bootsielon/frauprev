# === ml_pipeline/scaling.py ===========================================
"""
Step 4 – Scale numeric features (z‑score or median/IQR).

Spec compliance:
• Artefacts live in   artifacts/run_<global_hash>/scaling/
  (no hashes in filenames).
• Training computes centring / scaling stats; inference loads them and
  applies to new data (never recomputes from test).
• No self.hashes usage (legacy lines commented).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import mlflow
import numpy as np
import pandas as pd

from ml_pipeline.utils import log_registry  # absolute import – spec §11


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────
def standardize_dataframe(
    df: pd.DataFrame,
    numeric_cols: List[str],
    center_stats: pd.Series,
    scale_stats: pd.Series,
) -> pd.DataFrame:
    """
    Return a copy of *df* where numeric_cols have been standardised.

    Only columns that both (a) appear in numeric_cols and (b) are present in
    the provided dataframe are transformed.
    """
    df_scaled = df.copy()
    in_cols = [c for c in numeric_cols if c in df.columns]
    if in_cols:
        df_scaled[in_cols] = (df[in_cols] - center_stats[in_cols]) / scale_stats[in_cols]
    return df_scaled


# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE STEP
# ──────────────────────────────────────────────────────────────────────
def scaling(self) -> None:  # noqa: C901
    """
    Scale numeric features produced by *numeric_conversion*.

    • Training mode – compute stats on train set, save stats + scaled CSVs.  
    • Inference mode – load stats from training run, apply to test set
      (and any other provided splits) without recomputing.
    """
    step = "scaling"
    run_dir  = self.run_dir
    step_dir = os.path.join(run_dir, step)
    train_dir = os.path.join("artifacts", f"run_{self.global_train_hash}", step)

    os.makedirs(step_dir, exist_ok=True)
    manifest_fp = os.path.join(step_dir, "manifest.json")

    # ── 0. quick exit when artefacts already present for this run ─────
    if os.path.exists(manifest_fp):
        manifest = json.load(open(manifest_fp))
        for key, rel in manifest["outputs"].items():
            if rel is None or not key.endswith("_csv"):
                continue
            full = os.path.join(step_dir, rel) if not os.path.isabs(rel) else rel
            df_key = key.replace("_csv", "")
            self.dataframes[df_key] = pd.read_csv(full)
        self.paths[step]     = step_dir
        self.artifacts[step] = manifest["outputs"]
        print(f"[{step.upper()}] Skipping — artefacts already present at {step_dir}")
        return

    # shorthand
    cfg = self.config
    target_col, id_col = cfg["target_col"], cfg["id_col"]

    # ──────────────────────────────
    # 1. DATA PREP
    # ──────────────────────────────
    datasets: Dict[str, pd.DataFrame] = {}
    for split in ["train_num", "val_num", "test_num", "excluded_num"]:
        if split in self.dataframes:
            datasets[split.replace("_num", "")] = self.dataframes[split]

    # we always need test set
    test_df = datasets["test"]

    # ──────────────────────────────
    # 2‑A. TRAINING MODE
    # ──────────────────────────────
    if self.train_mode:
        train_df = datasets["train"]
        # columns to scale
        numeric_cols = [
            c
            for c in train_df.select_dtypes(include="number").columns
            if c not in {target_col, id_col}
        ]

        # stats
        # center_func = np.mean if cfg["t1"] else np.median
        # scale_func = np.std if cfg["s1"] else lambda x: np.subtract(*np.percentile(x, [75, 25]))

        # center_stats = train_df[numeric_cols].agg(center_func)
        # scale_stats = train_df[numeric_cols].agg(scale_func).replace(0, 1.0)
        # --- PATCH inside scaling() ------------------------------------------
        # Compute translation and scale stats from TRAIN ONLY  ────────────
        if cfg["t1"]:  # mean
            center_stats = train_df[numeric_cols].mean()
        else:          # median
            center_stats = train_df[numeric_cols].median()

        if cfg["s1"]:  # standard deviation
            scale_stats = train_df[numeric_cols].std(ddof=0)
        else:          # IQR
            scale_stats = train_df[numeric_cols].apply(
                lambda x: np.subtract(*np.percentile(x, [75, 25]))
            )

        scale_stats.replace(0, 1.0, inplace=True)  # avoid division by zero
        # ---------------------------------------------------------------------
        # scale every split present
        scaled: Dict[str, pd.DataFrame] = {}
        for name, df in datasets.items():
            scaled_df = standardize_dataframe(df, numeric_cols, center_stats, scale_stats)
            # put back untouched id/target (ensures exact originals)
            for col in [target_col, id_col]:
                if col in df.columns:
                    scaled_df[col] = df[col]
            scaled[name] = scaled_df

        # save csvs
        for name, df in scaled.items():
            df.to_csv(os.path.join(step_dir, f"{name}_sca.csv"), index=False)

        # save stats
        stats = {
            "center_function": "mean" if cfg["t1"] else "median",
            "scale_function": "std" if cfg["s1"] else "iqr",
            "center": center_stats.to_dict(),
            "scale": scale_stats.to_dict(),
            "numeric_cols": numeric_cols,
        }
        json.dump(stats, open(os.path.join(step_dir, "scaling_stats.json"), "w"), indent=2)

    # ──────────────────────────────
    # 2‑B. INFERENCE MODE
    # ──────────────────────────────
    else:
        stats_fp = os.path.join(train_dir, "scaling_stats.json")
        if not os.path.exists(stats_fp):
            raise FileNotFoundError(f"[{step}] scaling_stats.json missing in training run")

        stats = json.load(open(stats_fp))
        numeric_cols = stats["numeric_cols"]
        center_stats = pd.Series(stats["center"])
        scale_stats = pd.Series(stats["scale"]).replace(0, 1.0)

        # apply to *current* test data (and any other provided split)
        scaled = {}
        for name, df in datasets.items():
            scaled_df = standardize_dataframe(df, numeric_cols, center_stats, scale_stats)
            for col in [target_col, id_col]:
                if col in df.columns:
                    scaled_df[col] = df[col]
            scaled[name] = scaled_df
            scaled_df.to_csv(os.path.join(step_dir, f"{name}_sca.csv"), index=False)

        # copy stats file locally for completeness
        import shutil

        shutil.copy(stats_fp, os.path.join(step_dir, "scaling_stats.json"))

    # ──────────────────────────────
    # 3. MANIFEST & REGISTRY
    # ──────────────────────────────
    outputs = {
        f"{name}_sca_csv": f"{name}_sca.csv"
        for name in scaled.keys()
    }
    outputs["scaling_stats_json"] = "scaling_stats.json"

    manifest = {
        "step": step,
        "global_hash": self.global_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "output_dir": step_dir,
        "outputs": outputs,
    }
    json.dump(manifest, open(manifest_fp, "w"), indent=2)

    # MLflow
    if cfg.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{self.global_hash}"):
            mlflow.set_tags({"step": step, "global_hash": self.global_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    # registry + state
    log_registry(step, self.global_hash, cfg, step_dir)

    for name, df in scaled.items():
        self.dataframes[f"{name}_sca"] = df

    self.paths[step]     = step_dir
    self.artifacts[step] = outputs
    # removed: self.hashes no longer used

    print(f"[{step.upper()}] Scaling completed — artefacts at {step_dir}")


# ──────────────────────────────────────────────────────────────────────
# SMOKE‑TEST
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Synthetic smoke‑test that runs scaling directly after numeric_conversion.
    """
    from ml_pipeline.base import MLPipeline
    import numpy as np

    # --- create mock numeric‑dataframes (already numeric) -------------
    np.random.seed(0)
    n = 300
    df = pd.DataFrame(
        {
            "id": range(n),
            "f1": np.random.normal(10, 2, n),
            "f2": np.random.exponential(1, n),
            "target": np.random.choice([0, 1], n),
        }
    )
    cfg = {
        "target_col": "target",
        "id_col": "id",
        "t1": True,
        "s1": True,
        "use_mlflow": False,
        "seed": 42,
    }

    pipe = MLPipeline(cfg, data_source="raw", raw_data=df)
    pipe.dataframes = {"train_num": df, "val_num": df.sample(60), "test_num": df.sample(60)}
    pipe.scaling()

    print("Train mean after scaling (should be ≈0):\n", pipe.dataframes["train_sca"][["f1", "f2"]].mean())
    print("Artefacts →", pipe.paths["scaling"])
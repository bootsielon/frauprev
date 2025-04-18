# === ml_pipeline/eda.py ==============================================
"""
Step 0 – Exploratory Data Analysis (EDA)

• Uses the *global* `self.global_hash` created in base.py.
• Artifacts live in              artifacts/run_<hash>/eda/
• `self.train_mode`
      True   → create all artifacts (summary, metadata, class plot, sample)
      False  → re‑use artifacts from the training run and only create the
               lightweight `raw_sample.csv` if it does not exist.
• No duplicated code paths—artifact creation is implemented once and
  executed only when necessary.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns

from ml_pipeline.utils import log_registry


# ──────────────────────────────────────────────────────────────────────
def eda(self) -> None:
    """
    Perform EDA for the current run.

    Updates
    -------
    self.dataframes["raw"]
    self.paths["eda"]
    """

    step      = "eda"
    step_dir  = os.path.join(self.run_dir, step)
    manifest  = os.path.join(step_dir, "manifest.json")
    raw_csv   = os.path.join(step_dir, "raw_sample.csv")

    # ------------------------------------------------------------------
    # helper that builds *all* artifacts (runs only when needed)
    # ------------------------------------------------------------------
    def _build_artifacts(df: pd.DataFrame) -> dict[str, Any]:
        summary_file   = os.path.join(step_dir, "summary_stats.csv")
        metadata_file  = os.path.join(step_dir, "column_metadata.json")
        class_plot_png = os.path.join(step_dir, "class_distribution.png")

        # summary stats
        df.describe(include="all").to_csv(summary_file)

        # column metadata
        metadata = {
            col: {
                "dtype"      : str(df[col].dtype),
                "cardinality": int(df[col].nunique(dropna=False)),
                "nulls"      : int(df[col].isnull().sum()),
            }
            for col in df.columns
        }
        with open(metadata_file, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

        # class distribution plot (if target exists)
        target_col = self.config.get("target_col")
        if target_col and target_col in df.columns:
            plt.figure()
            sns.countplot(x=target_col, data=df)
            plt.title("Class Distribution")
            plt.savefig(class_plot_png, bbox_inches="tight")
            plt.close()
        else:
            class_plot_png = None
            print(f"[{step.upper()}] target_col '{target_col}' not found – skipping class plot.")

        return {
            "summary_stats_csv"      : os.path.basename(summary_file),
            "column_metadata_json"   : os.path.basename(metadata_file),
            "class_distribution_plot": os.path.basename(class_plot_png) if class_plot_png else None,
        }

    # ------------------------------------------------------------------
    # ensure directory exists (training always, inference maybe)
    # ------------------------------------------------------------------
    os.makedirs(step_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # load data
    # ------------------------------------------------------------------
    df = self.load_data()
    self.dataframes["raw"] = df

    # always write a tiny sample so later steps can reload quickly
    df.head(500).to_csv(raw_csv, index=False)

    # ------------------------------------------------------------------
    # decide whether to build heavy artifacts
    # ------------------------------------------------------------------
    if self.train_mode or not os.path.exists(manifest):
        outputs = _build_artifacts(df)
    else:
        # inference mode with existing manifest → read it back
        with open(manifest, "r", encoding="utf-8") as fp:
            manifest_data = json.load(fp)
        outputs = manifest_data["outputs"]

    # ------------------------------------------------------------------
    # write (or overwrite) manifest
    # ------------------------------------------------------------------
    manifest_data = {
        "step"      : step,
        "run_hash"  : self.global_hash,
        "timestamp" : datetime.now(timezone.utc).isoformat(),
        "train_mode": self.train_mode,
        "config"    : {
            "target_col": self.config.get("target_col"),
            "n_rows"    : len(df),
            "n_cols"    : df.shape[1],
        },
        "outputs"   : {
            **outputs,
            "raw_sample_csv": os.path.basename(raw_csv),
        },
    }
    with open(manifest, "w", encoding="utf-8") as fp:
        json.dump(manifest_data, fp, indent=2)

    # ------------------------------------------------------------------
    # optional MLflow logging (only in training)
    # ------------------------------------------------------------------
    if self.train_mode and self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{self.global_hash}"):
            mlflow.set_tags({"step": step, "run_hash": self.global_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    # ------------------------------------------------------------------
    # registry + bookkeeping
    # ------------------------------------------------------------------
    log_registry(step, self.global_hash, manifest_data["config"], step_dir)
    self.paths[step] = step_dir

    print(f"[{step.upper()}] Done – artifacts at {step_dir}")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Ad‑hoc manual test.

    Example:
        python -m ml_pipeline.eda            # training mode
        python -m ml_pipeline.eda infer <hash>  # inference mode
    """
    import sys
    from ml_pipeline.base import MLPipeline

    args = sys.argv[1:]
    mode = "train" if not args or args[0] != "infer" else "infer"

    cfg: dict[str, Any] = {
        "target_col": "is_fraud",
        "use_mlflow": False,
        "train_mode": mode == "train",
    }
    if mode == "infer":
        try:
            cfg["train_hash"] = args[1]
        except IndexError:
            raise SystemExit("Usage: python -m ml_pipeline.eda infer <train_hash>")

    pipe = MLPipeline(config=cfg)
    pipe.eda()
    print(f"EDA finished; outputs in {pipe.paths['eda']}")
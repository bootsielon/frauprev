# === ml_pipeline/eda.py ==============================================
"""
Step 0 – Exploratory Data Analysis (EDA)

• Uses the global `self.global_hash` set in base.py.
• Artefacts live in  artifacts/run_<hash>/eda/
• `self.train_mode`
      True  → build summary, metadata, class plot, and a raw head sample.
      False → re‑use training artefacts; create `raw_sample.csv` locally
               only if it is missing.
• Heavy computation occurs once per training run (skip‑guard – spec §14).
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
def eda(self) -> None:  # type: ignore[override]
    """
    Perform EDA for the current pipeline run.

    Side-effects
    ------------
    • Updates self.dataframes["raw"] with the loaded raw dataset.
    • Populates self.paths["eda"] with the directory holding EDA artefacts.
    """
    step = "eda"
    step_dir = os.path.join("artifacts", f"run_{self.global_hash}", step)
    manifest_fp = os.path.join(step_dir, "manifest.json")
    raw_csv = os.path.join(step_dir, "raw_sample.csv")

    train_step_dir = os.path.join("artifacts", f"run_{self.global_train_hash}", step)
    train_manifest_fp = os.path.join(train_step_dir, "manifest.json")

    # ──────────────────────────────────────────────────────────────────
    # Skip‑guard (spec §14)
    # ──────────────────────────────────────────────────────────────────
    if os.path.exists(manifest_fp):
        with open(manifest_fp, "r", encoding="utf-8") as fp:
            mdata = json.load(fp)
        self.paths[step] = step_dir
        log_registry(step, self.global_hash, mdata["config"], step_dir)
        print(f"[{step.upper()}] Skipped – artefacts already exist at {step_dir}")
        return

    if not self.train_mode:
        if not os.path.exists(train_manifest_fp):
            raise AssertionError(
                f"[{step.upper()}] Missing training artefacts at {train_step_dir}"
            )
        os.makedirs(step_dir, exist_ok=True)
        train_raw_csv = os.path.join(train_step_dir, "raw_sample.csv")
        if os.path.exists(train_raw_csv) and not os.path.exists(raw_csv):
            import shutil
            shutil.copy2(train_raw_csv, raw_csv)

        with open(train_manifest_fp, "r", encoding="utf-8") as fp:
            mdata = json.load(fp)
        self.paths[step] = train_step_dir
        log_registry(step, self.global_hash, mdata["config"], train_step_dir)
        print(f"[{step.upper()}] Re‑used training artefacts at {train_step_dir}")
        return

    # ──────────────────────────────────────────────────────────────────
    # Training mode – full EDA
    # ──────────────────────────────────────────────────────────────────
    os.makedirs(step_dir, exist_ok=True)

    df = self.load_data()
    self.dataframes["raw"] = df

    df.head(500).to_csv(raw_csv, index=False)

    summary_csv = os.path.join(step_dir, "summary_stats.csv")
    meta_json = os.path.join(step_dir, "column_metadata.json")
    class_png = os.path.join(step_dir, "class_distribution.png")

    df.describe(include="all").to_csv(summary_csv)

    metadata = {
        col: {
            "dtype": str(df[col].dtype),
            "cardinality": int(df[col].nunique(dropna=False)),
            "nulls": int(df[col].isnull().sum()),
        }
        for col in df.columns
    }
    with open(meta_json, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    target_col = self.config.get("target_col")
    if target_col and target_col in df.columns:
        plt.figure()
        sns.countplot(x=target_col, data=df)
        plt.title("Class Distribution")
        plt.savefig(class_png, bbox_inches="tight")
        plt.close()
        class_plot = os.path.basename(class_png)
    else:
        print(f"[{step.upper()}] target_col '{target_col}' not found – skipping class plot.")
        class_plot = None

    manifest = {
        "step": step,
        "param_hash": self.global_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "target_col": target_col,
            "n_rows": len(df),
            "n_cols": df.shape[1],
        },
        "output_dir": step_dir,
        "outputs": {
            "raw_sample_csv": os.path.basename(raw_csv),
            "summary_stats_csv": os.path.basename(summary_csv),
            "column_metadata_json": os.path.basename(meta_json),
            "class_distribution_plot": class_plot,
        },
    }

    with open(manifest_fp, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{self.global_hash}"):
            mlflow.set_tags({"step": step, "param_hash": self.global_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    self.paths[step] = step_dir
    log_registry(step, self.global_hash, manifest["config"], step_dir)

    print(f"[{step.upper()}] Done – artefacts at {step_dir}")


if __name__ == "__main__":
    """
    Smoke‑tests for ml_pipeline.eda  (spec §19)

    Scenarios exercised
    -------------------
    1. Training with no pre‑existing artefacts  (fresh run)
    2. Training with artefacts present          (skip‑guard hit)
    3. Inference with required artefacts        (reuse training outputs)
    4. Inference without artefacts              (must fail clearly)
    """

    import os
    import shutil
    import traceback
    import pandas as pd

    from ml_pipeline.base import MLPipeline
    from ml_pipeline.utils import DEFAULT_TEST_HASH

    # Clean slate
    # if os.path.exists("artifacts"):
        # shutil.rmtree("artifacts")

    df_demo = pd.DataFrame(
        {
            "client_id": [1, 2],
            "merchant_id": [10, 20],
            "amount": [100.5, 200.0],
            "timestamp": ["2023-01-01", "2023-01-02"],
            "is_fraud": [0, 1],
        }
    )

    # Shared config builder
    def make_cfg(train_mode: bool, **overrides):
        base = {
            "train_mode": train_mode,
            "model_name": "demo_model",
            "model_hash": "abc1234",
            "dataset_name": "demo_ds",
            "target_col": "is_fraud",
            "use_mlflow": False,
            "feature_names": ["amount", "timestamp"],
        }
        return {**base, **overrides}

    def safe(label: str, fn):
        try:
            fn()
            print(f"[OK ] {label}")
        except Exception as exc:
            print(f"[ERR] {label} → {exc}")
            traceback.print_exc()

    # 1. Training – fresh run
    cfg_train = make_cfg(True)
    pipe_1 = MLPipeline(cfg_train, data_source="raw", raw_data=df_demo)
    pipe_1.global_hash = DEFAULT_TEST_HASH
    pipe_1.global_train_hash = DEFAULT_TEST_HASH
    pipe_1.run_dir = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}")
    safe("TRAIN‑v1: EDA fresh", lambda: pipe_1.eda())

    # 2. Training – same config, skip‑guard
    pipe_2 = MLPipeline(cfg_train, data_source="raw", raw_data=df_demo)
    pipe_2.global_hash = DEFAULT_TEST_HASH
    pipe_2.global_train_hash = DEFAULT_TEST_HASH
    pipe_2.run_dir = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}")
    safe("TRAIN‑v2: EDA skip‑guard", lambda: pipe_2.eda())

    # 3. Inference – correct train_hash
    cfg_infer = make_cfg(False, train_hash=DEFAULT_TEST_HASH)
    pipe_3 = MLPipeline(cfg_infer, data_source="raw", raw_data=df_demo)
    pipe_3.global_hash = DEFAULT_TEST_HASH  # set explicitly
    pipe_3.global_train_hash = DEFAULT_TEST_HASH
    pipe_3.run_dir = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}")
    safe("INFER: EDA re‑use", lambda: pipe_3.eda())

    # 4. Inference – bad train_hash (must fail)
    bad_hash = "not_a_real_hash"
    cfg_fail = make_cfg(False, train_hash=bad_hash)
    cfg_fail["model_name"] = "new_model"  # force a new inference hash
    cfg_fail["model_hash"] = "zzzz9999"
    pipe_4 = MLPipeline(cfg_fail, data_source="raw", raw_data=df_demo)
    pipe_4.global_hash = "badf00ddead0"  # override inference run hash
    pipe_4.global_train_hash = bad_hash
    pipe_4.run_dir = os.path.join("artifacts", f"run_{pipe_4.global_hash}")
    try:
        pipe_4.eda()
        raise SystemExit("[TEST] Expected failure not raised")
    except AssertionError as e:
        print(f"[OK ] INFER‑fail: Correctly failed → {e}")

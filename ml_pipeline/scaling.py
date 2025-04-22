"""
Step 4 – Scaling / Standardisation
==================================

Applies centring and scaling to *numeric* columns using **train‑only**
statistics and writes the results to

    artifacts/run_<self.global_hash>/scaling/

Spec compliance highlights
──────────────────────────
• One global hash per run – no per‑step hashes (SPEC §1, §2).  
• Skip‑guard is the very first runtime check (SPEC §14).  
• Inference: reuse → load‑from‑train → raise (SPEC §5).  
• Filenames carry **no hashes** because the folder already embeds it
  (SPEC §25).  
• `self.hashes` removed; `log_registry` called (SPEC §3, §7).  
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import mlflow

from ml_pipeline.utils import convert_numpy_types, log_registry


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def _standardise(
    df: pd.DataFrame,
    numeric_cols: List[str],
    centre: pd.Series,
    scale: pd.Series,
) -> pd.DataFrame:
    """
    Return a copy of *df* where *numeric_cols* are centred/scaled.

    Non‑numeric or excluded columns are left untouched.
    """
    df_out = df.copy()
    if numeric_cols:
        df_out[numeric_cols] = (df[numeric_cols] - centre[numeric_cols]) / scale[numeric_cols]
    return df_out


def _load_existing(step_dir: str) -> Dict[str, pd.DataFrame]:
    """Load <split>_scaled.csv files that already exist in *step_dir*."""
    dfs: Dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test", "excluded"):
        fp = os.path.join(step_dir, f"{split}_scaled.csv")
        if os.path.exists(fp):
            dfs[f"{split}_sca"] = pd.read_csv(fp)
    return dfs


# --------------------------------------------------------------------------- #
# Main pipeline step                                                          #
# --------------------------------------------------------------------------- #
def scaling(self) -> None:  # type: ignore[override]
    """
    Standardise numeric variables using train‑only statistics.

    Behaviour
    ---------
    • Training:
        – Compute centre/scale stats.
        – Apply to train/val/test(/excluded).
        – Persist artefacts + manifest.
    • Inference:
        – Reuse artefacts in *current* run if present.
        – Else load from training run.
        – Else raise FileNotFoundError.
    """
    step = "scaling"

    # ------------------------------------------------------------------- #
    # Paths / filenames                                                   #
    # ------------------------------------------------------------------- #
    run_step_dir = os.path.join("artifacts", f"run_{self.global_hash}", step)
    run_manifest = os.path.join(run_step_dir, "manifest.json")

    # ------------------------------------------------------------------- #
    # 0️⃣  Skip‑guard – artefacts already in *current* run                #
    # ------------------------------------------------------------------- #
    if os.path.exists(run_manifest):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {run_step_dir}")
        self.paths[step] = run_step_dir
        self.dataframes.update(_load_existing(run_step_dir))
        return

    # ------------------------------------------------------------------- #
    # 1️⃣  Inference → load artefacts from the training run               #
    # ------------------------------------------------------------------- #
    if not self.train_mode:
        train_step_dir = os.path.join("artifacts", f"run_{self.global_train_hash}", step)
        train_manifest = os.path.join(train_step_dir, "manifest.json")
        if os.path.exists(train_manifest):
            print(f"[{step.upper()}] Reusing training artefacts from {train_step_dir}")
            self.paths[step] = train_step_dir
            self.dataframes.update(_load_existing(train_step_dir))
            return
        raise FileNotFoundError(
            f"[{step.upper()}] Expected training artefacts at {train_step_dir} but none found."
        )

    # ------------------------------------------------------------------- #
    # 2️⃣  Training mode – compute and persist                            #
    # ------------------------------------------------------------------- #
    cfg = self.config
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    train_df = self.dataframes["train_num"]
    val_df = self.dataframes["val_num"]
    test_df = self.dataframes["test_num"]
    excluded_df = self.dataframes.get("excluded_num")

    target_col = cfg["target_col"]
    id_col = cfg["id_col"]
    exclude_cols = [c for c in (target_col, id_col) if c in train_df.columns]

    numeric_cols = [
        c
        for c in train_df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    # ------------------------------------------------ centre / scale ----
    centre_func = np.mean if cfg.get("t1", True) else np.median
    scale_func = np.std if cfg.get("s1", True) else lambda x: np.subtract(*np.percentile(x, [75, 25]))

    centre_stats = train_df[numeric_cols].agg(centre_func)
    scale_stats = train_df[numeric_cols].agg(scale_func).replace(0, 1.0)

    # ------------------------------------------------ transform ---------
    train_scaled = _standardise(train_df, numeric_cols, centre_stats, scale_stats)
    val_scaled = _standardise(val_df, numeric_cols, centre_stats, scale_stats)
    test_scaled = _standardise(test_df, numeric_cols, centre_stats, scale_stats)
    excluded_scaled = (
        _standardise(excluded_df, numeric_cols, centre_stats, scale_stats)
        if excluded_df is not None
        else None
    )

    # ------------------------------------------------ persist artefacts -
    os.makedirs(run_step_dir, exist_ok=True)

    out_files: Dict[str, str] = {
        "train_scaled_csv": os.path.join(run_step_dir, "train_scaled.csv"),
        "val_scaled_csv": os.path.join(run_step_dir, "val_scaled.csv"),
        "test_scaled_csv": os.path.join(run_step_dir, "test_scaled.csv"),
        "scaling_stats_json": os.path.join(run_step_dir, "scaling_stats.json"),
    }
    if excluded_scaled is not None:
        out_files["excluded_scaled_csv"] = os.path.join(run_step_dir, "excluded_scaled.csv")

    train_scaled.to_csv(out_files["train_scaled_csv"], index=False)
    val_scaled.to_csv(out_files["val_scaled_csv"], index=False)
    test_scaled.to_csv(out_files["test_scaled_csv"], index=False)
    if excluded_scaled is not None:
        excluded_scaled.to_csv(out_files["excluded_scaled_csv"], index=False)

    stats_payload: Dict[str, Any] = {
        "centre_function": "mean" if cfg.get("t1", True) else "median",
        "scale_function": "std" if cfg.get("s1", True) else "iqr",
        "centre": convert_numpy_types(centre_stats.to_dict()),
        "scale": convert_numpy_types(scale_stats.to_dict()),
        "numeric_columns": numeric_cols,
        "excluded_columns": exclude_cols,
    }
    with open(out_files["scaling_stats_json"], "w") as fh:
        json.dump(stats_payload, fh, indent=2)

    manifest = {
        "step": step,
        "param_hash": self.global_hash,  # still recorded for traceability
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "t1": cfg.get("t1", True),
            "s1": cfg.get("s1", True),
            "seed": seed,
        },
        "output_dir": run_step_dir,
        "outputs": {k: os.path.basename(v) for k, v in out_files.items()},
    }
    with open(run_manifest, "w") as fh:
        json.dump(manifest, fh, indent=2)

    # ------------------------------------------------ MLflow / registry -
    if cfg.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{self.global_hash}"):
            mlflow.set_tags({"step": step, "global_hash": self.global_hash})
            mlflow.log_artifacts(run_step_dir, artifact_path=step)

    log_registry(step, self.global_hash, manifest["config"], run_step_dir)

    # ------------------------------------------------ update pipeline ---
    self.dataframes.update(
        {
            "train_sca": train_scaled,
            "val_sca": val_scaled,
            "test_sca": test_scaled,
        }
    )
    if excluded_scaled is not None:
        self.dataframes["excluded_sca"] = excluded_scaled

    self.paths[step] = run_step_dir
    self.artifacts[step] = out_files

    print(
        f"[{step.upper()}] Done – artefacts at {run_step_dir}  "
        f"(train {len(train_scaled)}, val {len(val_scaled)}, test {len(test_scaled)})"
    )


if __name__ == "__main__":
    """
    Smoke‑tests for ml_pipeline.scaling  (SPEC §19)

    Scenarios exercised
    -------------------
    1. Training with no pre‑existing artefacts  (fresh run)
    2. Training with artefacts present          (skip‑guard hit)
    3. Inference with required artefacts        (reuse training outputs)
    4. Inference without artefacts              (must fail clearly)

    All tests rely on the shared DEFAULT_TEST_HASH to guarantee artefact
    continuity across pipeline steps (SPEC §17).
    """
    import os
    import shutil
    import traceback
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from ml_pipeline.base import MLPipeline
    from ml_pipeline.utils import DEFAULT_TEST_HASH

    step = "scaling"

    # ------------------------------------------------------------------ #
    # Helper                                                             #
    # ------------------------------------------------------------------ #
    def build_cfg(train_mode: bool, **kw) -> dict:
        cfg: dict = {
            "train_mode": train_mode,
            "model_name": "dummy_model",
            "model_hash": "abcd1234",
            "dataset_name": "dummy_ds",
            "feature_names": ["f1", "f2"],
            "target_col": "target",
            "id_col": "id",
            "seed": 123,
            "t1": True,   # mean
            "s1": True,   # std
            "use_mlflow": False,
            # mandatory for inference
            "train_hash": DEFAULT_TEST_HASH,
        }
        cfg.update(kw)
        return cfg

    def safe(label: str, fn):
        try:
            fn()
            print(f"[OK ] {label}")
        except Exception as exc:
            print(f"[ERR] {label} → {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Mock numeric‑only data (as produced by numeric_conversion)         #
    # ------------------------------------------------------------------ #
    np.random.seed(42)
    n = 120
    mock_train = pd.DataFrame(
        {
            "id": range(1, n + 1),
            "feature_A": np.random.normal(10, 2, n),
            "feature_B": np.random.normal(50, 5, n),
            "target": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        }
    )
    mock_val = mock_train.sample(30, random_state=1).reset_index(drop=True)
    mock_test = mock_train.sample(30, random_state=2).reset_index(drop=True)
    mock_excluded = mock_train.sample(10, random_state=3).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Clean slate for DEFAULT_TEST_HASH                                  #
    # ------------------------------------------------------------------ #
    artefact_root = Path("artifacts") / f"run_{DEFAULT_TEST_HASH}" / step
    if artefact_root.exists():
        shutil.rmtree(artefact_root)

    # 1️⃣  Training – fresh artefacts
    pipe_train = MLPipeline(build_cfg(True))
    pipe_train.global_hash = DEFAULT_TEST_HASH
    pipe_train.global_train_hash = DEFAULT_TEST_HASH
    pipe_train.dataframes = {
        "train_num": mock_train,
        "val_num": mock_val,
        "test_num": mock_test,
        "excluded_num": mock_excluded,
    }
    print("\n>>> TRAINING RUN (fresh artefacts)")
    safe("TRAIN‑fresh", pipe_train.scaling)

    # 2️⃣  Training – skip‑guard hit
    pipe_train_skip = MLPipeline(build_cfg(True))
    pipe_train_skip.global_hash = DEFAULT_TEST_HASH
    pipe_train_skip.global_train_hash = DEFAULT_TEST_HASH
    pipe_train_skip.dataframes = {
        "train_num": mock_train,
        "val_num": mock_val,
        "test_num": mock_test,
    }
    print("\n>>> TRAINING RUN (should skip)")
    safe("TRAIN‑skip‑guard", pipe_train_skip.scaling)

    # 3️⃣  Inference – artefacts present
    infer_hash_ok = "abcdefabcdef"
    pipe_infer_ok = MLPipeline(build_cfg(False, train_hash=DEFAULT_TEST_HASH))
    pipe_infer_ok.global_hash = infer_hash_ok
    pipe_infer_ok.global_train_hash = DEFAULT_TEST_HASH
    pipe_infer_ok.dataframes = {
        "test_num": mock_test
    }  # only need test for this step
    print("\n>>> INFERENCE RUN (artefacts present)")
    safe("INFER‑reuse", pipe_infer_ok.scaling)

    # 4️⃣  Inference – artefacts missing (expect failure)
    missing_train_hash = "feedfeedfeed"
    artefact_miss_dir = Path("artifacts") / f"run_{missing_train_hash}" / step
    if artefact_miss_dir.exists():
        shutil.rmtree(artefact_miss_dir)

    pipe_infer_fail = MLPipeline(build_cfg(False, train_hash=missing_train_hash))
    pipe_infer_fail.global_hash = "deadbeef0000"
    pipe_infer_fail.global_train_hash = missing_train_hash
    print("\n>>> INFERENCE RUN (artefacts missing – should fail)")
    try:
        pipe_infer_fail.scaling()
        print("❌  ERROR: Missing‑artefact inference did *not* fail as expected")
    except FileNotFoundError as e:
        print(f"✅  Caught expected error → {e}")
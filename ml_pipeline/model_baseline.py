"""
Step 5 – Baseline Model
=======================

Trains an XGBoost baseline classifier (or re‑uses an existing one) and
stores artefacts in

    artifacts/run_<self.global_hash>/model_baseline/

Spec compliance highlights
──────────────────────────
• No per‑step hashes; folder already embeds the run‑hash (SPEC §1‑§2, §25).  
• Skip‑guard is the first runtime check (SPEC §14).  
• Inference logic: reuse → load‑from‑train → raise (SPEC §5).  
• `self.hashes` removed; `log_registry` invoked (SPEC §3, §7).  
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc,
)

from ml_pipeline.utils import log_registry, save_plot_as_artifact


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def _load_existing(step_dir: str) -> Dict[str, Any]:
    """Load model + metrics previously saved in *step_dir*."""
    artefacts = {}
    model_fp = os.path.join(step_dir, "model.json")
    if os.path.exists(model_fp):
        model = XGBClassifier()
        model.load_model(model_fp)
        artefacts["model"] = model

    metrics_fp = os.path.join(step_dir, "metrics.json")
    if os.path.exists(metrics_fp):
        with open(metrics_fp) as fh:
            artefacts["metrics"] = json.load(fh)

    features_fp = os.path.join(step_dir, "feature_names.json")
    if os.path.exists(features_fp):
        with open(features_fp) as fh:
            meta = json.load(fh)
            if "model" in artefacts:
                artefacts["model"].feature_names = meta["feature_names"]
            artefacts["feature_names"] = meta["feature_names"]

    return artefacts


# --------------------------------------------------------------------------- #
# Main pipeline step                                                          #
# --------------------------------------------------------------------------- #
def model_baseline(self) -> None:  # type: ignore[override]
    """
    Train or load the baseline XGBoost classifier.
    """
    step = "model_baseline"
    run_step_dir = os.path.join("artifacts", f"run_{self.global_hash}", step)
    manifest_fp = os.path.join(run_step_dir, "manifest.json")

    # ------------------------------------------------------------------- #
    # 0️⃣  Skip‑guard – artefacts already in *current* run                #
    # ------------------------------------------------------------------- #
    if os.path.exists(manifest_fp):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {run_step_dir}")
        artefacts = _load_existing(run_step_dir)
        self.models["baseline"] = artefacts["model"]
        self.metrics["baseline"] = artefacts["metrics"]
        self.paths[step] = run_step_dir
        self.artifacts[step] = {
            "model_file": os.path.join(run_step_dir, "model.json"),
            "metrics_file": os.path.join(run_step_dir, "metrics.json"),
            "feature_names_file": os.path.join(run_step_dir, "feature_names.json"),
        }
        return

    # ------------------------------------------------------------------- #
    # 1️⃣  Inference → load artefacts from the training run               #
    # ------------------------------------------------------------------- #
    if not self.train_mode:
        train_step_dir = os.path.join("artifacts", f"run_{self.global_train_hash}", step)
        train_manifest = os.path.join(train_step_dir, "manifest.json")
        if os.path.exists(train_manifest):
            print(f"[{step.upper()}] Reusing training artefacts from {train_step_dir}")
            artefacts = _load_existing(train_step_dir)
            self.models["baseline"] = artefacts["model"]
            self.metrics["baseline"] = artefacts["metrics"]
            self.paths[step] = train_step_dir
            self.artifacts[step] = {
                "model_file": os.path.join(train_step_dir, "model.json"),
                "metrics_file": os.path.join(train_step_dir, "metrics.json"),
                "feature_names_file": os.path.join(train_step_dir, "feature_names.json"),
            }
            return
        raise FileNotFoundError(
            f"[{step.upper()}] Expected training artefacts at {train_step_dir} but none found."
        )

    # ------------------------------------------------------------------- #
    # 2️⃣  Training mode – compute and persist                            #
    # ------------------------------------------------------------------- #
    cfg = self.config
    seed = cfg.get("random_state", 42)
    np.random.seed(seed)

    targ = cfg["target_col"]

    # -------------------- fetch scaled data generated earlier ----------
    train_df = self.dataframes["train_sca"]
    val_df = self.dataframes["val_sca"]
    test_df = self.dataframes["test_sca"]
    excl_df = self.dataframes.get("excluded_sca")

    # drop potential non‑numeric leftovers
    drop_obj = [c for c in train_df.columns if train_df[c].dtype == "object"]
    if drop_obj:
        print(f"[{step.upper()}] Dropping object columns: {drop_obj}")
        train_df = train_df.drop(columns=drop_obj)
        val_df = val_df.drop(columns=drop_obj)
        test_df = test_df.drop(columns=drop_obj)
        if excl_df is not None:
            excl_df = excl_df.drop(columns=drop_obj)

    feature_cols = [c for c in train_df.columns if c != targ]

    X_train, y_train = train_df[feature_cols], train_df[targ]
    X_val, y_val = val_df[feature_cols], val_df[targ]
    X_test, y_test = test_df[feature_cols], test_df[targ]
    if excl_df is not None:
        X_excl, y_excl = excl_df[feature_cols], excl_df[targ]

    # ------------------------------- train model -----------------------
    model = XGBClassifier(
        n_estimators=cfg.get("n_estimators", 400),
        max_depth=cfg.get("max_depth", 4),
        learning_rate=cfg.get("learning_rate", 0.01),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        random_state=seed,
        use_label_encoder=False,
        eval_metric="auc",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    model.feature_names = feature_cols  # keep explicit list

    # ------------------------------- metrics ---------------------------
    def _roc(y_true, y_prob):
        return roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5

    metrics = {
        "train": {
            "accuracy": accuracy_score(y_train, model.predict(X_train)),
            "roc_auc": _roc(y_train, model.predict_proba(X_train)[:, 1]),
        },
        "val": {
            "accuracy": accuracy_score(y_val, model.predict(X_val)),
            "roc_auc": _roc(y_val, model.predict_proba(X_val)[:, 1]),
        },
        "test": {
            "accuracy": accuracy_score(y_test, model.predict(X_test)),
            "roc_auc": _roc(y_test, model.predict_proba(X_test)[:, 1]),
        },
    }
    if excl_df is not None:
        metrics["excluded"] = {
            "accuracy": accuracy_score(y_excl, model.predict(X_excl)),
            "roc_auc": _roc(y_excl, model.predict_proba(X_excl)[:, 1]),
        }

    # ------------------------------- persist artefacts -----------------
    os.makedirs(run_step_dir, exist_ok=True)
    model_fp = os.path.join(run_step_dir, "model.json")
    metrics_fp = os.path.join(run_step_dir, "metrics.json")
    features_fp = os.path.join(run_step_dir, "feature_names.json")

    model.save_model(model_fp)
    with open(metrics_fp, "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(features_fp, "w") as fh:
        json.dump({"feature_names": feature_cols}, fh, indent=2)

    # feature importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    ax.bar(range(len(importances)), importances[order])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(np.array(feature_cols)[order], rotation=45, ha="right")
    ax.set_title("Feature Importances")
    fig.tight_layout()
    fi_png = os.path.join(run_step_dir, "feature_importance.png")
    save_plot_as_artifact(fig, fi_png, {}, "dummy")  # dict not used further

    # ROC curve (test set)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], "--")
    ax_roc.set_xlabel("FPR"), ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC – Test")
    ax_roc.legend()
    roc_png = os.path.join(run_step_dir, "roc_curve.png")
    save_plot_as_artifact(fig_roc, roc_png, {}, "dummy2")

    manifest = {
        "step": step,
        "param_hash": self.global_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            k: cfg.get(k)
            for k in (
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "random_state",
            )
        },
        "output_dir": run_step_dir,
        "outputs": {
            "model_file": os.path.basename(model_fp),
            "metrics_file": os.path.basename(metrics_fp),
            "feature_names_file": os.path.basename(features_fp),
            "feature_importance_png": os.path.basename(fi_png),
            "roc_curve_png": os.path.basename(roc_png),
        },
        "metrics": metrics,
    }
    with open(manifest_fp, "w") as fh:
        json.dump(manifest, fh, indent=2)

    # ------------------------------- MLflow / registry -----------------
    if cfg.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{self.global_hash}"):
            mlflow.set_tags({"step": step, "global_hash": self.global_hash})
            mlflow.log_artifacts(run_step_dir, artifact_path=step)
            flat_metrics = {
                f"{ds}_{m}": v
                for ds, mset in metrics.items()
                for m, v in mset.items()
                if v is not None
            }
            mlflow.log_metrics(flat_metrics)

    log_registry(step, self.global_hash, manifest["config"], run_step_dir)

    # ------------------------------- update pipeline -------------------
    self.models["baseline"] = model
    self.metrics["baseline"] = metrics
    self.paths[step] = run_step_dir
    self.artifacts[step] = manifest["outputs"]

    print(
        f"[{step.upper()}] Done – artefacts at {run_step_dir}  "
        f"(test AUC {metrics['test']['roc_auc']:.3f})"
    )


if __name__ == "__main__":
    """
    Smoke‑tests for ml_pipeline.model_baseline  (SPEC §19)

    Fix: indices were reset *before* the .drop() calls, triggering a KeyError.
    We now keep the original indices for drop‑based splitting and reset them
    only afterwards.
    """
    import os
    import shutil
    import traceback

    import numpy as np
    import pandas as pd

    from ml_pipeline.base import MLPipeline
    from ml_pipeline.utils import DEFAULT_TEST_HASH

    step = "model_baseline"

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
            "random_state": 123,
            "use_mlflow": False,
            "n_estimators": 50,
        }
        if not train_mode:
            cfg["train_hash"] = kw.get("train_hash")
        return cfg

    def safe(label: str, fn):
        try:
            fn()
            print(f"[OK ] {label}")
        except Exception as exc:
            print(f"[ERR] {label} → {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Create deterministic numeric data                                  #
    # ------------------------------------------------------------------ #
    np.random.seed(42)
    n = 150
    mock_df = pd.DataFrame(
        {
            "id": range(1, n + 1),
            "f1": np.random.normal(0, 1, n),
            "f2": np.random.normal(5, 2, n),
            "target": np.random.choice([0, 1], n, p=[0.75, 0.25]),
        }
    )

    # simple split – keep indices intact for `.drop()`
    train_df = mock_df.sample(frac=0.6, random_state=1)
    val_df = mock_df.drop(train_df.index).sample(frac=0.4, random_state=2)
    test_df = mock_df.drop(train_df.index).drop(val_df.index)
    excl_df = mock_df.sample(20, random_state=3)

    # finally reset indices
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    excl_df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------ #
    # Clean slate                                                         #
    # ------------------------------------------------------------------ #
    artefact_root = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}", step)
    if os.path.exists(artefact_root):
        shutil.rmtree(artefact_root)

    # 1️⃣  Training – fresh artefacts
    pipe_train = MLPipeline(build_cfg(True))
    pipe_train.global_hash = DEFAULT_TEST_HASH
    pipe_train.global_train_hash = DEFAULT_TEST_HASH
    pipe_train.dataframes = {
        "train_sca": train_df,
        "val_sca": val_df,
        "test_sca": test_df,
        "excluded_sca": excl_df,
    }
    print("\n>>> TRAINING RUN (fresh artefacts)")
    safe("TRAIN‑fresh", pipe_train.model_baseline)

    # 2️⃣  Training – skip‑guard
    pipe_train_skip = MLPipeline(build_cfg(True))
    pipe_train_skip.global_hash = DEFAULT_TEST_HASH
    pipe_train_skip.global_train_hash = DEFAULT_TEST_HASH
    pipe_train_skip.dataframes = {
        "train_sca": train_df,
        "val_sca": val_df,
        "test_sca": test_df,
    }
    print("\n>>> TRAINING RUN (should skip)")
    safe("TRAIN‑skip‑guard", pipe_train_skip.model_baseline)

    # 3️⃣  Inference – artefacts present
    infer_hash_ok = "abcabcabcabc"
    pipe_infer_ok = MLPipeline(build_cfg(False, train_hash=DEFAULT_TEST_HASH))
    pipe_infer_ok.global_hash = infer_hash_ok
    pipe_infer_ok.global_train_hash = DEFAULT_TEST_HASH
    pipe_infer_ok.dataframes = {"test_sca": test_df}
    print("\n>>> INFERENCE RUN (artefacts present)")
    safe("INFER‑reuse", pipe_infer_ok.model_baseline)

    # 4️⃣  Inference – artefacts missing (should fail)
    missing_train_hash = "feedfeedfeed"
    miss_dir = os.path.join("artifacts", f"run_{missing_train_hash}", step)
    if os.path.exists(miss_dir):
        shutil.rmtree(miss_dir)

    pipe_infer_fail = MLPipeline(build_cfg(False, train_hash=missing_train_hash))
    pipe_infer_fail.global_hash = "deadbeef0000"
    pipe_infer_fail.global_train_hash = missing_train_hash
    print("\n>>> INFERENCE RUN (artefacts missing – should fail)")
    try:
        pipe_infer_fail.model_baseline()
        print("❌  ERROR: Missing‑artefact inference did *not* fail as expected")
    except FileNotFoundError as e:
        print(f"✅  Caught expected error → {e}")
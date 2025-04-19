# === ml_pipeline/model_baseline.py ====================================
"""
Step 5 – Train (or load) a baseline XGBoost model and log metrics.

Spec compliance
• artefacts live in  artifacts/run_<global_hash>/model_baseline/
• no filenames contain hashes
• self.hashes is no longer used
• inference reuses the training artefacts; no re‑training, no duplication
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Tuple

import joblib  # still handy for quick dumps
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from ml_pipeline.utils import log_registry, save_plot_as_artifact  # absolute imports



# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────
def _train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict,
) -> XGBClassifier:
    """Return a fitted XGBClassifier with `feature_names` attribute."""
    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        random_state=config["random_state"],
        use_label_encoder=False,
        eval_metric="auc",
    )
    model.fit(X_train, y_train)
    model.feature_names = list(X_train.columns)
    return model


def _dataset_metrics(
    model: XGBClassifier, X: pd.DataFrame, y: pd.Series
) -> Tuple[float, float]:
    """Return (accuracy, roc_auc). If only one class present, roc_auc=0.5."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, y_pred)
    roc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.5
    return acc, roc



# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE STEP
# ──────────────────────────────────────────────────────────────────────
def model_baseline(self) -> None:  # noqa: C901
    step = "model_baseline"

    # ------------------------------------------------------------------
    # 0. Common paths
    # ------------------------------------------------------------------
    run_dir  = self.run_dir
    step_dir = os.path.join(run_dir, step)
    train_dir = os.path.join("artifacts", f"run_{self.global_train_hash}", step)
    os.makedirs(step_dir, exist_ok=True)
    manifest_fp = os.path.join(step_dir, "manifest.json")

    # ------------------------------------------------------------------
    # 1. Fast‑exit if artefacts already present for *this* run
    # ------------------------------------------------------------------
    if os.path.exists(manifest_fp):
        manifest = json.load(open(manifest_fp))
        model = XGBClassifier()
        model.load_model(os.path.join(step_dir, manifest["outputs"]["model_file"]))
        with open(os.path.join(step_dir, manifest["outputs"]["feature_names_file"])) as fh:
            model.feature_names = json.load(fh)["feature_names"]

        self.models["baseline"] = model
        self.metrics["baseline"] = json.load(
            open(os.path.join(step_dir, manifest["outputs"]["metrics_file"]))
        )
        self.paths[step]     = step_dir
        self.artifacts[step] = manifest["outputs"]
        print(f"[{step.upper()}] Skipping — artefacts already present at {step_dir}")
        return

    # ------------------------------------------------------------------
    # 2. Configuration & data
    # ------------------------------------------------------------------
    cfg = {
        "n_estimators": self.config.get("n_estimators", 400),
        "max_depth": self.config.get("max_depth", 4),
        "learning_rate": self.config.get("learning_rate", 0.01),
        "subsample": self.config.get("subsample", 0.8),
        "colsample_bytree": self.config.get("colsample_bytree", 0.8),
        "random_state": self.config.get("random_state", 42),
    }
    targ = self.config["target_col"]

    # load scaled splits created in Step 4
    splits = {n.replace("_sca", ""): df for n, df in self.dataframes.items() if n.endswith("_sca")}

    # Ensure consistent feature set (exclude target & any non numeric cols)
    base_df = splits["train"].drop(columns=targ)
    feature_names = list(base_df.columns)
    for name, df in splits.items():
        splits[name] = df[feature_names + [targ]]  # reorder / subset

    # ------------------------------------------------------------------
    # 3‑A. TRAINING MODE
    # ------------------------------------------------------------------
    if self.train_mode:
        X_train, y_train = base_df, splits["train"][targ]
        model = _train_xgb(X_train, y_train, cfg)

        # metrics on all available splits
        metrics = {}
        for name, df in splits.items():
            X, y = df.drop(columns=targ), df[targ]
            acc, roc = _dataset_metrics(model, X, y)
            metrics[name] = {"accuracy": acc, "roc_auc": roc}

        # ── save artefacts ────────────────────────────────────────────
        model_fp   = os.path.join(step_dir, "model.json")
        metrics_fp = os.path.join(step_dir, "metrics.json")
        feats_fp   = os.path.join(step_dir, "feature_names.json")

        model.save_model(model_fp)
        json.dump(metrics, open(metrics_fp, "w"), indent=2)
        json.dump({"feature_names": feature_names}, open(feats_fp, "w"), indent=2)

        # feature‑importance plot
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(importances)), importances[idx])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(np.array(feature_names)[idx], rotation=45, ha="right")
        ax.set_title("Feature Importances")
        fig.tight_layout()
        fi_png = os.path.join(step_dir, "feature_importance.png")
        save_plot_as_artifact(fig, fi_png, {}, "feature_importance_png")

    # ------------------------------------------------------------------
    # 3‑B. INFERENCE MODE
    # ------------------------------------------------------------------
    else:
        # copy model + feature list from training run
        model_src = os.path.join(train_dir, "model.json")
        feats_src = os.path.join(train_dir, "feature_names.json")
        if not os.path.exists(model_src):
            raise FileNotFoundError(f"[{step}] Training model missing at {model_src}")

        import shutil

        shutil.copy(model_src, step_dir)
        shutil.copy(os.path.join(train_dir, "feature_importance.png"), step_dir)
        shutil.copy(os.path.join(train_dir, "metrics.json"), step_dir)

        model = XGBClassifier()
        model.load_model(os.path.join(step_dir, "model.json"))
        model.feature_names = json.load(open(os.path.join(step_dir, "feature_names.json")))["feature_names"]

        # metrics for current inference data (usually just 'test')
        metrics = {}
        for name, df in splits.items():
            X, y = df.drop(columns=targ), df[targ]
            acc, roc = _dataset_metrics(model, X, y)
            metrics[name] = {"accuracy": acc, "roc_auc": roc}

        # overwrite metrics.json with the new ones (keep training metrics separately)
        json.dump(metrics, open(os.path.join(step_dir, "metrics.json"), "w"), indent=2)

    # ------------------------------------------------------------------
    # 4. Manifest, MLflow, registry, state update
    # ------------------------------------------------------------------
    outputs = {
        "model_file": "model.json",
        "metrics_file": "metrics.json",
        "feature_names_file": "feature_names.json",
        "feature_importance_png": "feature_importance.png",
    }

    manifest = {
        "step": step,
        "global_hash": self.global_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "output_dir": step_dir,
        "outputs": outputs,
        "metrics": metrics,
    }
    json.dump(manifest, open(manifest_fp, "w"), indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{self.global_hash}"):
            mlflow.set_tags({"step": step, "global_hash": self.global_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)
            mlflow.log_metrics(
                {
                    f"{ds}_{m}": v
                    for ds, md in metrics.items()
                    for m, v in md.items()
                }
            )

    log_registry(step, self.global_hash, cfg, step_dir)

    # pipeline state
    self.models["baseline"] = model
    self.metrics["baseline"] = metrics
    self.paths[step] = step_dir
    self.artifacts[step] = outputs
    print(f"[{step.upper()}] Completed — artefacts at {step_dir}")


# ──────────────────────────────────────────────────────────────────────
# SMOKE‑TEST
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Synthetic smoke‑test: builds fake numeric data, pretends scaling ran,
    then executes model_baseline in *training* mode.
    """
    from ml_pipeline.base import MLPipeline

    np.random.seed(0)
    n = 800
    df = pd.DataFrame(
        {
            "id": range(n),
            "f1": np.random.normal(0, 1, n),
            "f2": np.random.normal(0, 1, n),
            "target": np.random.binomial(1, 0.3, n),
        }
    )
    cfg = {
        "target_col": "target",
        "id_col": "id",
        "use_mlflow": False,
        "seed": 42,
    }

    pipe = MLPipeline(cfg, data_source="raw", raw_data=df)
    pipe.dataframes = {
        "train_sca": df.iloc[:500].copy(),
        "val_sca": df.iloc[500:650].copy(),
        "test_sca": df.iloc[650:].copy(),
    }
    pipe.scaling = lambda: None  # stub, not needed here
    pipe.run_dir = os.path.join("artifacts", f"run_{pipe.global_hash}")
    os.makedirs(pipe.run_dir, exist_ok=True)

    pipe.model_baseline()
    print("Metrics:", pipe.metrics["baseline"]["test"])
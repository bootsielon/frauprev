import os
import json
import joblib
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from xgboost import XGBClassifier
import mlflow
from .utils import make_param_hash
from .utils import log_registry


def hyperparameter_tuning(self) -> None:
    """
    Step 11: Run K-fold CV on validation set to tune hyperparameters.
    Use only selected features and metric specified in config.
    Store all config-metric results and the best config.
    """
    step = "hyperparameter_tuning"
    df_val = self.dataframes["val_scaled"]
    features = (
        self.dataframes.get("cluster_selection_metrics", {})
        or self.dataframes["threshold_selection_metrics"]
    )
    optimal_config_row = features.sort_values(
        by=f"val_{self.config['opt_metric']}",
        ascending=not self.config["minimize_metric"]
    ).iloc[0]
    selected_features = optimal_config_row["features"]

    k = self.config["k_folds"]
    metric_key = self.config["opt_metric"]
    minimize = self.config["minimize_metric"]

    default_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8]
    }
    grid = self.config.get("param_grid", default_grid)

    param_list = list(itertools.product(
        grid["n_estimators"],
        grid["max_depth"],
        grid["learning_rate"],
        grid["subsample"],
        grid["colsample_bytree"]
    ))

    config = {
        "selected_features_hash": self.hashes["feature_select_cluster"]
        if "cluster_selection_metrics" in self.dataframes
        else self.hashes["feature_select_threshold"],
        "k_folds": k,
        "opt_metric": metric_key,
        "minimize_metric": minimize,
        "param_grid": grid
    }

    param_hash = make_param_hash(config)
    step_dir = os.path.join("artifacts", f"{step}_{param_hash}")
    manifest_path = os.path.join(step_dir, "manifest.json")

    if os.path.exists(manifest_path):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = param_hash
        self.dataframes["cv_tuning_results"] = pd.read_csv(
            os.path.join(step_dir, f"cv_tuning_results_{param_hash}.csv"))
        return

    os.makedirs(step_dir, exist_ok=True)

    X = df_val[selected_features]
    y = df_val[self.config["target_col"]]
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.config["seed"])

    def compute_metrics(y_true, y_pred, y_prob):
        auc = roc_auc_score(y_true, y_prob)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return {
            "auroc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sensitivity": sensitivity,
            "specificity": specificity
        }

    records = []
    for params in param_list:
        cfg = {
            "n_estimators": params[0],
            "max_depth": params[1],
            "learning_rate": params[2],
            "subsample": params[3],
            "colsample_bytree": params[4],
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": self.config["seed"]
        }

        scores = []
        for train_idx, val_idx in cv.split(X, y):
            model = XGBClassifier(**cfg)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            probs = model.predict_proba(X.iloc[val_idx])[:, 1]
            score = compute_metrics(y.iloc[val_idx], preds, probs)[metric_key]
            scores.append(score)

        record = {
            "params": cfg,
            "metric_values": scores,
            "metric_avg": np.mean(scores),
            "metric_min": np.min(scores)
        }
        records.append(record)

    results_df = pd.DataFrame(records)
    results_df.to_csv(os.path.join(step_dir, f"cv_tuning_results_{param_hash}.csv"), index=False)

    best_row = results_df.loc[results_df["metric_min" if minimize else "metric_avg"].idxmax()]
    self.models["best_hyperparams"] = best_row["params"]

    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "cv_results_csv": f"cv_tuning_results_{param_hash}.csv"
        }
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    log_registry(step, param_hash, config, step_dir)

    self.dataframes["cv_tuning_results"] = results_df
    self.paths[step] = step_dir
    self.hashes[step] = param_hash
    # self.models[step] = best_row["params"]
    # self.metrics[step] = { "train": best_row["metric_avg"], "val": best_row["metric_avg"], "test": None, "holdout": None}
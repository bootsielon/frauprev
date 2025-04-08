import os
import json
import joblib
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # type: ignore
# import seaborn as sns  # type: ignore
from datetime import datetime
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix  # , roc_curve
)
from xgboost import XGBClassifier
import mlflow
from .utils import make_param_hash
from .utils import log_registry


def feature_select_cluster(self) -> None:
    """
    Step 10A: Feature selection via hierarchical clustering.
    For each cluster count (1 to N_features):
        - cluster correlation matrix
        - pick top SHAP feature per cluster
        - train/evaluate model, store metrics + features used
    """
    step = "feature_select_cluster"
    df_corr = self.dataframes["feature_correlation"]
    shap_df = self.dataframes["shap_normalized"].mean()
    target_col = self.config["target_col"]
    id_col = self.config["id_col"]

    config = {
        "correlation_hash": self.hashes["feature_correlation"],
        "shap_hash": self.hashes["shap_selection"],
        "save_fs_mods": self.config.get("save_fs_mods", False),
        "model_type": "xgboost",
        "hyperparams": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": self.config["seed"]
        }
    }

    param_hash = make_param_hash(config)
    step_dir = os.path.join("artifacts", f"{step}_{param_hash}")
    manifest_file = os.path.join(step_dir, "manifest.json")

    if os.path.exists(manifest_file):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = param_hash
        self.dataframes["cluster_selection_metrics"] = pd.read_csv(
            os.path.join(step_dir, f"metrics_cluster_select_{param_hash}.csv"))
        return

    os.makedirs(step_dir, exist_ok=True)

    # Prep training + evaluation data
    df_train = self.dataframes["train_scaled"]
    df_val = self.dataframes["val_scaled"]
    df_test = self.dataframes["test_scaled"]
    df_holdout = self.dataframes["excluded_majority"]

    def prepare(df, cols):
        X = df[cols]
        y = df[target_col]
        return X, y

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

    corr_values = 1 - df_corr.abs()
    linkage_matrix = linkage(corr_values, method="average")

    results = []
    shap_scores = shap_df.to_dict()

    for k in range(1, len(df_corr.columns) + 1):
        cluster_ids = fcluster(linkage_matrix, k, criterion="maxclust")
        cluster_df = pd.DataFrame({"feature": df_corr.columns, "cluster": cluster_ids})
        selected = (
            cluster_df.groupby("cluster")
            .apply(lambda grp: grp["feature"].map(shap_scores).idxmax())
            .values
        )
        selected = list(selected)

        X_train, y_train = prepare(df_train, selected)
        X_val, y_val = prepare(df_val, selected)
        X_test, y_test = prepare(df_test, selected)
        X_holdout, y_holdout = prepare(df_holdout, selected)

        model = XGBClassifier(**config["hyperparams"])
        model.fit(X_train, y_train)

        metrics = {
            "k": k,
            "features": selected,
            "train": compute_metrics(y_train, model.predict(X_train), model.predict_proba(X_train)[:, 1]),
            "val": compute_metrics(y_val, model.predict(X_val), model.predict_proba(X_val)[:, 1]),
            "test": compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1]),
            "holdout": compute_metrics(y_holdout, model.predict(X_holdout), model.predict_proba(X_holdout)[:, 1]),
        }

        if config["save_fs_mods"]:
            model_path = os.path.join(step_dir, f"model_k{k}_{param_hash}.joblib")
            joblib.dump(model, model_path)
            metrics["model_path"] = model_path

        results.append(metrics)

    # Flatten + store
    records = []
    for entry in results:
        base = {"k": entry["k"], "features": entry["features"]}
        for split in ["train", "val", "test", "holdout"]:
            for metric, value in entry[split].items():
                base[f"{split}_{metric}"] = value
        records.append(base)

    df_results = pd.DataFrame(records)
    df_results.to_csv(os.path.join(step_dir, f"metrics_cluster_select_{param_hash}.csv"), index=False)

    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "metrics_csv": f"metrics_cluster_select_{param_hash}.csv"
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    log_registry(step, param_hash, config, step_dir)

    self.dataframes["cluster_selection_metrics"] = df_results
    self.paths[step] = step_dir
    self.hashes[step] = param_hash
    # self.models[step] = model
    # self.dataframes["feature_selection"] = df_results[["k", "features"]].copy()
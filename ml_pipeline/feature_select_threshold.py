import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import mlflow
from .utils import make_param_hash
from .utils import log_registry
from .feature_select_cluster import feature_select_cluster


def get_features(threshold: float, feature_order: list, abs_corr: pd.DataFrame, shap_df: pd.Series) -> list:
    """
    Select features based on correlation thresholding.
    For each pair of features with correlation above the threshold,
    keep the one with higher SHAP value.

    Args:
        threshold (float): Correlation threshold.
        feature_order (list): Ordered list of features based on SHAP values.
        abs_corr (pd.DataFrame): Absolute correlation matrix.
        shap_df (pd.Series): SHAP values for features.

        Returns:
            list: List of selected features.
    """
    keep = set(feature_order)
    for i, f1 in enumerate(feature_order):
        for f2 in feature_order[i+1:]:
            if f2 not in keep or f1 not in keep:
                continue
            if abs_corr.loc[f1, f2] > threshold:
                keep.discard(f2 if shap_df[f1] >= shap_df[f2] else f1)
    return sorted(keep)

def prepare(self, df, cols):
    """
    Prepare the data for model training and evaluation.
    This includes selecting features and separating the target variable.
    
    Args:
        df (pd.DataFrame): The input dataframe containing features and target.
        cols (list): List of feature column names.

        
    Returns:
        tuple: A tuple containing the feature matrix (X) and target vector (y).

    """
    X = df[cols]
    y = df[self.config["target_col"]]
    return X, y

def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute various evaluation metrics for binary classification.
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray): Predicted probabilities.


    Returns:
        dict: Dictionary containing computed metrics.
    """
    
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


def run_feature_selection(self, stage: int) -> None:
    """
    Wrapper for running Stage 0 or 1 feature selection using clustering or thresholding.
    """
    use_cluster = self.config["use_cluster_select"][stage]
    save_mods = self.config.get("save_fs_mods", False)

    if stage == 0:
        base_features = list(self.dataframes["shap_normalized"].columns)
        corr_matrix = self.dataframes["feature_correlation"]
        hyperparams = self.config["baseline_hyperparams"]
    else:
        base_features = list(self.models["best_hyperparams"]["features"])
        corr_matrix = self.dataframes["feature_correlation"].loc[base_features, base_features]
        hyperparams = self.models["best_hyperparams"]["params"]

    if use_cluster:
        feature_select_cluster(self, base_features, corr_matrix, hyperparams, stage)
    else:
        feature_select_threshold(self, base_features, corr_matrix, hyperparams, stage)


def feature_select_threshold(self) -> None:
    """
    Step 10B: Feature selection using a decreasing correlation threshold.
    At each threshold:
        - Drop one feature from any correlated pair with abs(corr) > threshold
        - Keep the one with higher SHAP value
        - Train model, store metrics, features, and optionally model
    """
    step = "feature_select_threshold"
    df_corr = self.dataframes["feature_correlation"]
    shap_df = self.dataframes["shap_normalized"].mean()
    abs_corr = df_corr.abs()
    feature_order = shap_df.sort_values(ascending=False).index.tolist()
    max_corr = abs_corr.where(np.triu(np.ones(abs_corr.shape), 1).astype(bool)).max().max()

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
        },
        "threshold_step": 0.01,
        "min_threshold": 0.5
    }

    param_hash = make_param_hash(config)
    step_dir = os.path.join("artifacts", f"{step}_{param_hash}")
    manifest_path = os.path.join(step_dir, "manifest.json")

    if os.path.exists(manifest_path):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = param_hash
        self.dataframes["threshold_selection_metrics"] = pd.read_csv(
            os.path.join(step_dir, f"metrics_threshold_select_{param_hash}.csv"))
        return

    os.makedirs(step_dir, exist_ok=True)

    df_train = self.dataframes["train_scaled"]
    df_val = self.dataframes["val_scaled"]
    df_test = self.dataframes["test_scaled"]
    df_holdout = self.dataframes["excluded_majority"]

    epsilon = 1e-6
    thresholds = [round(th, 6) for th in np.arange(max_corr + epsilon, config["min_threshold"] - 1e-5, -config["threshold_step"])]

    results = []

    for th in thresholds:
        selected = get_features(th)

        X_train, y_train = prepare(df_train, selected)
        X_val, y_val = prepare(df_val, selected)
        X_test, y_test = prepare(df_test, selected)
        X_holdout, y_holdout = prepare(df_holdout, selected)

        model = XGBClassifier(**config["hyperparams"])
        model.fit(X_train, y_train)

        metrics = {
            "threshold": th,
            "features": selected,
            "train": compute_metrics(y_train, model.predict(X_train), model.predict_proba(X_train)[:, 1]),
            "val": compute_metrics(y_val, model.predict(X_val), model.predict_proba(X_val)[:, 1]),
            "test": compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1]),
            "holdout": compute_metrics(y_holdout, model.predict(X_holdout), model.predict_proba(X_holdout)[:, 1])
        }

        if config["save_fs_mods"]:
            model_path = os.path.join(step_dir, f"model_th_{th:.4f}_{param_hash}.joblib")
            joblib.dump(model, model_path)
            metrics["model_path"] = model_path

        results.append(metrics)

    records = []
    for entry in results:
        base = {"threshold": entry["threshold"], "features": entry["features"]}
        for split in ["train", "val", "test", "holdout"]:
            for metric, value in entry[split].items():
                base[f"{split}_{metric}"] = value
        records.append(base)

    df_results = pd.DataFrame(records)
    df_results.to_csv(os.path.join(step_dir, f"metrics_threshold_select_{param_hash}.csv"), index=False)

    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "metrics_csv": f"metrics_threshold_select_{param_hash}.csv"
        }
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    log_registry(step, param_hash, config, step_dir)

    self.dataframes["threshold_selection_metrics"] = df_results
    self.paths[step] = step_dir
    self.hashes[step] = param_hash
    # self.models[step] = model if config["save_fs_mods"] else None
    # self.metrics[step] = { "train": df_results["train_auroc"].max(), "val": df_results["val_auroc"].max(), "test": df_results["test_auroc"].max(), "holdout": df_results["holdout_auroc"].max()}
    # self.metrics[step] = { "train": df_results["train_auroc"].max(), "val": df_results["val_auroc"].max(), "test": df_results["test_auroc"].max(), "holdout": df_results["holdout_auroc"].max()}
import os
import json
import joblib
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import mlflow
from .utils import make_param_hash
from .utils import log_registry


def final_model(self) -> None:
    """
    Step 13: Train final model with optimal features + hyperparameters.
    Evaluate on all splits. Save model, metrics, plots.
    """
    step = "final_model"
    features = self.models["best_hyperparams"]["features"]
    hyperparams = self.models["best_hyperparams"]["params"]

    config = {
        "final_features_hash": self.hashes["feature_select_cluster_stage1"]
        if "cluster_selection_metrics" in self.dataframes
        else self.hashes["feature_select_threshold_stage1"],
        "hyperparam_hash": self.hashes["hyperparameter_tuning"],
        "model_type": "xgboost",
        "hyperparams": hyperparams
    }

    param_hash = make_param_hash(config)
    step_dir = os.path.join("artifacts", f"{step}_{param_hash}")
    manifest_path = os.path.join(step_dir, "manifest.json")

    if os.path.exists(manifest_path):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = param_hash
        self.models["final_model"] = joblib.load(os.path.join(step_dir, f"final_model_{param_hash}.joblib"))
        return

    os.makedirs(step_dir, exist_ok=True)

    def prepare(df):
        return df[features], df[self.config["target_col"]]

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

    # Data splits
    df_train = self.dataframes["train_scaled"]
    df_val = self.dataframes["val_scaled"]
    df_test = self.dataframes["test_scaled"]
    df_holdout = self.dataframes["excluded_majority"]

    X_train, y_train = prepare(df_train)
    X_val, y_val = prepare(df_val)
    X_test, y_test = prepare(df_test)
    X_holdout, y_holdout = prepare(df_holdout)

    model = XGBClassifier(**hyperparams)
    model.fit(X_train, y_train)

    def plot_roc(y_true, y_prob, split):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{split} ROC")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{split} ROC Curve")
        plt.legend()
        path = os.path.join(step_dir, f"roc_{split}_{param_hash}.png")
        plt.savefig(path)
        plt.close()
        return path

    results = {}
    for name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
        ("holdout", X_holdout, y_holdout)
    ]:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        results[name] = compute_metrics(y, preds, probs)
        results[name]["roc_path"] = plot_roc(y, probs, name)

    results_df = pd.DataFrame(results).T
    results_csv = os.path.join(step_dir, f"final_metrics_{param_hash}.csv")
    results_df.to_csv(results_csv)

    model_path = os.path.join(step_dir, f"final_model_{param_hash}.joblib")
    joblib.dump(model, model_path)

    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "model_path": model_path,
            "metrics_csv": results_csv,
            "roc_curves": {k: v["roc_path"] for k, v in results.items()}
        }
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(results_csv)
            for path in manifest["outputs"]["roc_curves"].values():
                mlflow.log_artifact(path)

    log_registry(step, param_hash, config, step_dir)

    self.models["final_model"] = model
    self.dataframes["final_metrics"] = results_df
    self.paths[step] = step_dir
    self.hashes[step] = param_hash
    # self.models["final_model_path"] = model_path
    # self.models["final_model_metrics"] = results_df
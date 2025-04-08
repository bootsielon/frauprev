import os
import json
import joblib
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from datetime import datetime, timezone
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score
)
import mlflow
from .utils import make_param_hash
from .utils import log_registry


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute classification metrics for binary classification.
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


def model_baseline(self) -> None:
    """
    Step 5: Train baseline model and evaluate performance.
    
    Updates:
        self.models["baseline"]
        self.metrics["baseline"]
        self.paths["model_baseline"]
        self.hashes["model_baseline"]
    """
    step = "model_baseline"
    
    # Load scaled data from previous step
    train_df = self.dataframes["train_sca"]
    val_df = self.dataframes["val_sca"]
    test_df = self.dataframes["test_sca"]
    excluded_df = self.dataframes.get("excluded_sca", None)

    # Diagnostic information
    print(f"[{step.upper()}] Checking scaled dataframes...")
    print(f"  - Target column name: '{self.config['target_col']}'")
    # print(f"  - Train columns: {train_df.columns.tolist()}")
    # print(f"  - Did scaling step run from scratch? {self.artifacts.get('scaling', {}).get('train_scaled_csv') is not None}")
    
    print(f"  - Train dataset shape: {train_df.shape}")
    print(f"  - Validation dataset shape: {val_df.shape}")
    print(f"  - Test dataset shape: {test_df.shape}")
    if excluded_df is not None:
        print(f"  - Holdout dataset shape: {excluded_df.shape}")
    else:
        print("  - No holdout dataset available.")
    targname = self.config['target_col']
    # Print available dataframe keys for debugging
    print(f"[{step.upper()}] Target value investigation:")
    print(f"  - Train dataset target values: {train_df[targname].unique().tolist()}")
    print(f"  - Validation dataset target values: {val_df[targname].unique().tolist()}")
    print(f"  - Test dataset target values: {test_df[targname].unique().tolist()}")
    
    # Check if excluded data exists (just once)
    if excluded_df is not None:
        print(f"  - Holdout dataset target values: {excluded_df[targname].unique().tolist()}")
    else:
        print("  - No holdout dataset available.")

    # Print original target values
    print(f"  - Original feature_engineered target values: {self.dataframes['feature_engineered'][targname].unique().tolist()}")
    
    # Print previous step info
    print(f"  - Previous step: numeric_conversion, path: {self.paths['numeric_conversion']}")
    # print(f"  - numeric_conversion config: {self.transformations['numeric_conversion']['grouping_map'].keys()}")
    
    # Basic configuration
    config = {
        "n_estimators": self.config.get("n_estimators", 100),
        "max_depth": self.config.get("max_depth", 3),
        "learning_rate": self.config.get("learning_rate", 0.1),
        "subsample": self.config.get("subsample", 0.8),
        "colsample_bytree": self.config.get("colsample_bytree", 0.8),
        "random_state": self.config.get("random_state", 42)
    }
    
    param_hash = make_param_hash(config)
    step_dir = os.path.join("artifacts", f"{step}_{param_hash}")
    manifest_file = os.path.join(step_dir, "manifest.json")
    
    if os.path.exists(manifest_file):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = param_hash
        # Load model and metrics
        # TODO: Implement model loading
        self.models["baseline"] = joblib.load(os.path.join(step_dir, "model.pkl"))
        self.metrics["baseline"] = json.load(open(os.path.join(step_dir, "metrics.json")))
        return
    
    os.makedirs(step_dir, exist_ok=True)
    
    # Prepare data
    target_values = train_df[targname].unique()
    if set(target_values) != {0, 1}:
        print(f"[{step.upper()}] Warning: Non-standard target values detected: {target_values}. Transforming to 0/1 format for XGBoost.")
        # Map target values to 0/1
        target_mapping = {val: i for i, val in enumerate(target_values)}
        train_df[targname] = train_df[targname].map(target_mapping)
        val_df[targname] = val_df[targname].map(target_mapping)
        test_df[targname] = test_df[targname].map(target_mapping)
        excluded_df[targname] = excluded_df[targname].map(target_mapping) if excluded_df is not None else None
    
    # Print class distribution
    print(f"[{step.upper()}] Class distribution in training set: {dict(train_df[targname].value_counts())}")
    
    # Prepare X and y
    X_train = train_df.drop(targname, axis=1)
    y_train = train_df[targname]

    X_val = val_df.drop(targname, axis=1)
    y_val = val_df[targname]

    X_test = test_df.drop(targname, axis=1)
    y_test = test_df[targname]
    
    if excluded_df is not None:
        X_excluded = excluded_df.drop(targname, axis=1)
        y_excluded = excluded_df[targname]
    else:  
        X_excluded = None
        y_excluded = None

    # Handle object columns that XGBoost can't process
    object_columns = [col for col in X_train.columns if X_train[col].dtype == 'object']
    if object_columns:
        print(f"[{step.upper()}] Dropping object columns for XGBoost compatibility: {object_columns}")
        X_train = X_train.drop(columns=object_columns)
        X_val = X_val.drop(columns=object_columns)
        X_test = X_test.drop(columns=object_columns)
        X_excluded = X_excluded.drop(columns=object_columns) if excluded_df is not None else None
        # Convert object columns to categorical if needed
        if excluded_df is not None:
            X_excluded = excluded_df.drop(targname, axis=1)
            X_excluded = X_excluded.drop(columns=object_columns, errors='ignore')
    else:
        if excluded_df is not None:
            X_excluded = excluded_df.drop(targname, axis=1)
            y_excluded = excluded_df[targname]
        else:
            X_excluded = None
            y_excluded = None

    # Store feature names for consistency
    feature_names = X_train.columns.tolist()
    feature_names_file = os.path.join(step_dir, f"feature_names_{param_hash}.json")
    with open(feature_names_file, "w") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)

    # Ensure all datasets have the same columns in the same order
    X_train = X_train[feature_names]
    X_val = X_val[feature_names]  # self.ensure_column_consistency(X_val, feature_names)
    X_test = X_test[feature_names]  # self.ensure_column_consistency(X_test, feature_names)
    X_excluded = X_excluded[feature_names] if excluded_df is not None else None  # self.ensure_column_consistency(X_excluded, feature_names) if excluded_df is not None else None
    # X_excluded = self.ensure_column_consistency(X_excluded, feature_names) if excluded_df is not None else None
    # Ensure excluded_df has the same columns in the same order
    # if excluded_df is not None:
        # X_excluded = self.ensure_column_consistency(X_excluded, feature_names)
    
    # IMPORTANT FIX: When training, we need to make sure we use the SAME EXACT columns for train/test/val/excluded
    # Get the list of feature columns from the transformations that were stored in the numeric_conversion step
    """    if "feature_columns" in self.transformations["numeric_conversion"]:
            feature_columns = self.transformations["numeric_conversion"]["feature_columns"]
            feature_columns = [col for col in feature_columns if col != targname]
        else:
            # Fall back to just using the train columns if not explicitly stored
            feature_columns = [col for col in X_train.columns if col != targname]
        
        # Ensure all datasets have exactly the same columns in the same order
        X_train = X_train[feature_columns]
        X_val = val_df[feature_columns]
        X_test = test_df[feature_columns]
        X_excluded = excluded_df[feature_columns] if excluded_df is not None else None
    """    
    # if excluded_df is not None:
        # X_excluded = excluded_df[feature_columns]
    
    # Train model
    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        random_state=config["random_state"],
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Set feature_names explicitly to avoid mismatch errors
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    # Store feature names in the model
    model.feature_names = feature_names
    
    # Evaluate model
    print(f"[{step.upper()}] Target value mapping: {target_mapping if 'target_mapping' in locals() else 'No mapping needed'}")
    
    # Make predictions - ensure feature consistency!
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    
    y_pred_val = model.predict(X_val)
    y_prob_val = model.predict_proba(X_val)[:, 1]
    
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    y_pred_excluded = model.predict(X_excluded) if excluded_df is not None else None
    y_prob_excluded = model.predict_proba(X_excluded)[:, 1] if excluded_df is not None else None

    # Calculate metrics
    metrics = {
        "train": {
            "accuracy": accuracy_score(y_train, y_pred_train),
            "roc_auc": roc_auc_score(y_train, y_prob_train) if len(np.unique(y_train)) > 1 else 0.5
        },
        "val": {
            "accuracy": accuracy_score(y_val, y_pred_val),
            "roc_auc": roc_auc_score(y_val, y_prob_val) if len(np.unique(y_val)) > 1 else 0.5
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_pred_test),
            "roc_auc": roc_auc_score(y_test, y_prob_test) if len(np.unique(y_test)) > 1 else 0.5
        },
        "excluded": { #None  # Placeholder for excluded metrics
            "accuracy": accuracy_score(y_excluded, y_pred_excluded) if excluded_df is not None else None,
            "roc_auc": roc_auc_score(y_excluded, y_prob_excluded) if len(np.unique(y_excluded)) > 1 else 0.5 if excluded_df is not None else None
        }

    }
    
    """    
    if excluded_df is not None:
        y_excluded = excluded_df["target"]
        y_pred_excluded = model.predict(X_excluded)
        
        # Only calculate probabilistic metrics if we have both classes
        if len(np.unique(y_excluded)) > 1:
            y_prob_excluded = model.predict_proba(X_excluded)[:, 1]
            metrics["excluded"] = {
                "accuracy": accuracy_score(y_excluded, y_pred_excluded),
                "roc_auc": roc_auc_score(y_excluded, y_prob_excluded)
            }
        else:
            metrics["excluded"] = {
                "accuracy": accuracy_score(y_excluded, y_pred_excluded),
                "roc_auc": "N/A - only one class present"
            }
    """

    # Save model and metrics
    model_file = os.path.join(step_dir, f"model_{param_hash}.json")
    metrics_file = os.path.join(step_dir, f"metrics_{param_hash}.json")
    
    # Save model using built-in save method
    model.save_model(model_file)
    
    # Save metrics to JSON
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create and save manifest
    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "model_file": model_file,
            "metrics_file": metrics_file,
            "feature_names_file": feature_names_file
        },
        "metrics": metrics
    }
    
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)
            mlflow.log_metrics({f"{dataset}_{metric}": value 
                             for dataset, metrics_dict in metrics.items() 
                             for metric, value in metrics_dict.items() 
                             if not isinstance(value, str)})
    
    log_registry(step, param_hash, config, step_dir)
    
    # Update pipeline state
    self.models["baseline"] = model
    self.metrics["baseline"] = metrics
    self.paths[step] = step_dir
    self.hashes[step] = param_hash
    self.artifacts[step] = manifest["outputs"]
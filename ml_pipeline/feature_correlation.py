import os
import json
import pandas as pd
from datetime import datetime
import mlflow
from .utils import make_param_hash
from .utils import log_registry


def feature_correlation(self) -> None:
    """
    Step 9: Compute correlation matrix of SHAP-selected features.
    Use validation or training set depending on config["corr_val"].

    Updates:
        self.dataframes["feature_correlation"]
        self.paths["feature_correlation"], self.hashes["feature_correlation"]
    """
    step = "feature_correlation"
    use_validation = self.config.get("corr_val", True)
    df = self.dataframes["val_scaled"] if use_validation else self.dataframes["train_scaled"]

    selected_features = self.dataframes["shap_selected_features"]["feature"].tolist()
    df_selected = df[selected_features]

    config = {
        "selected_features_hash": self.hashes["shap_selection"],
        "corr_val": use_validation
    }
    param_hash = make_param_hash(config)
    step_dir = os.path.join("artifacts", f"{step}_{param_hash}")
    manifest_path = os.path.join(step_dir, "manifest.json")

    if os.path.exists(manifest_path):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = param_hash
        self.dataframes["feature_correlation"] = pd.read_csv(os.path.join(step_dir, f"correlation_matrix_{param_hash}.csv"), index_col=0)
        return

    os.makedirs(step_dir, exist_ok=True)

    corr_matrix = df_selected.corr(method="pearson")
    corr_csv = os.path.join(step_dir, f"correlation_matrix_{param_hash}.csv")
    corr_matrix.to_csv(corr_csv)

    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "correlation_matrix_csv": f"correlation_matrix_{param_hash}.csv"
        }
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    log_registry(step, param_hash, config, step_dir)

    self.dataframes["feature_correlation"] = corr_matrix
    self.paths[step] = step_dir
    self.hashes[step] = param_hash

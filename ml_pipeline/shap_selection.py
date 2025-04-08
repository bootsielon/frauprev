import os
import json
import pandas as pd
from datetime import datetime
import mlflow
from .utils import make_param_hash
from .utils import log_registry


def shap_selection(self) -> None:
    """
    Step 8: Normalize SHAP values, compute cumulative importance,
    and select features up to a cutoff percentage (sc1).

    Updates:
        self.dataframes["shap_normalized"]
        self.dataframes["shap_selected_features"]
        self.paths["shap_selection"], self.hashes["shap_selection"]
    """
    step = "shap_selection"
    df_shap = self.dataframes["shap_values"]

    config = {
        "sc1": self.config["sc1"]  # cumulative SHAP cutoff
    }
    param_hash = make_param_hash(config)
    step_dir = os.path.join("artifacts", f"{step}_{param_hash}")
    manifest_file = os.path.join(step_dir, "manifest.json")

    if os.path.exists(manifest_file):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = param_hash
        self.dataframes["shap_normalized"] = pd.read_csv(os.path.join(step_dir, f"shap_normalized_{param_hash}.csv"))
        self.dataframes["shap_selected_features"] = pd.read_csv(
            os.path.join(step_dir, f"shap_top_features_{param_hash}.csv")
        )
        return

    os.makedirs(step_dir, exist_ok=True)

    shap_abs = df_shap.abs()
    shap_sum = shap_abs.sum()
    shap_normalized = shap_abs / shap_sum.sum()
    shap_ranking = shap_normalized.mean().sort_values(ascending=False)

    cumulative = shap_ranking.cumsum()
    selected = cumulative[cumulative <= config["sc1"]].index.tolist()

    # Store
    df_shap_norm = shap_normalized
    df_top = pd.DataFrame({
        "feature": shap_ranking.index,
        "mean_shap": shap_ranking.values,
        "cumulative": cumulative.values
    })

    norm_path = os.path.join(step_dir, f"shap_normalized_{param_hash}.csv")
    top_path = os.path.join(step_dir, f"shap_top_features_{param_hash}.csv")

    df_shap_norm.to_csv(norm_path, index=False)
    df_top.to_csv(top_path, index=False)

    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "shap_normalized_csv": f"shap_normalized_{param_hash}.csv",
            "shap_top_features_csv": f"shap_top_features_{param_hash}.csv"
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    log_registry(step, param_hash, config, step_dir)

    self.dataframes["shap_normalized"] = df_shap_norm
    self.dataframes["shap_selected_features"] = df_top
    self.paths[step] = step_dir
    self.hashes[step] = param_hash

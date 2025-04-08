import os
import json
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import mlflow
from .utils import make_param_hash
from .utils import log_registry


def shap_explainability(self) -> None:
    """
    Step 6: Compute SHAP values for baseline model on training set.
    Stores SHAP values, plots, manifest, and updates pipeline state.

    Updates:
        self.dataframes["shap_values"]
        self.paths["shap_explainability"], self.hashes["shap_explainability"]
    """
    step = "shap_explainability"
    model_path = self.paths["model_baseline"]
    model_hash = self.hashes["model_baseline"]
    model = joblib.load(os.path.join(model_path, f"model_{model_hash}.joblib"))

    df_train = self.dataframes["train_scaled"]
    target_col = self.config["target_col"]
    id_col = self.config["id_col"]

    config = {
        "model_hash": model_hash
    }
    param_hash = make_param_hash(config)
    step_dir = os.path.join("artifacts", f"{step}_{param_hash}")
    manifest_file = os.path.join(step_dir, "manifest.json")

    if os.path.exists(manifest_file):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = param_hash
        self.dataframes["shap_values"] = pd.read_csv(os.path.join(step_dir, f"shap_values_{param_hash}.csv"))
        return

    os.makedirs(step_dir, exist_ok=True)

    # SHAP on training data
    X_train = df_train.drop(columns=[target_col, id_col], errors="ignore")

    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)

    shap_df = pd.DataFrame(shap_values.values, columns=X_train.columns)
    shap_df.to_csv(os.path.join(step_dir, f"shap_values_{param_hash}.csv"), index=False)

    # Bar Plot
    shap.plots.bar(shap_values, show=False)
    bar_path = os.path.join(step_dir, f"shap_bar_{param_hash}.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()

    # Beeswarm
    shap.plots.beeswarm(shap_values, show=False)
    beeswarm_path = os.path.join(step_dir, f"shap_beeswarm_{param_hash}.png")
    plt.savefig(beeswarm_path, bbox_inches="tight")
    plt.close()

    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "shap_csv": f"shap_values_{param_hash}.csv",
            "bar_plot": f"shap_bar_{param_hash}.png",
            "beeswarm_plot": f"shap_beeswarm_{param_hash}.png"
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    log_registry(step, param_hash, config, step_dir)

    self.dataframes["shap_values"] = shap_df
    self.paths[step] = step_dir
    self.hashes[step] = param_hash
    # print(f"[{step.upper()}] Completed and saved SHAP values and plots at {step_dir}")
    # print(f"[{step.upper()}] Manifest file saved at {manifest_file}")
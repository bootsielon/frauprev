import os
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from .utils import make_param_hash
from .utils import log_registry


def eda(self) -> None:
    """
    Step 0: Perform EDA. Saves summary statistics, column metadata, and class distribution plot.
    Updates: self.dataframes["raw"], self.paths["eda"], self.hashes["eda"]
    """
    df = self.load_data()
    self.dataframes["raw"] = df

    config = {
        "target_col": self.config["target_col"],
        "columns": sorted(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }
    param_hash = make_param_hash(config)
    step_key = "eda"
    step_dir = os.path.join("artifacts", f"{step_key}_{param_hash}")

    if os.path.exists(os.path.join(step_dir, "manifest.json")):
        print(f"[{step_key.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step_key] = step_dir
        self.hashes[step_key] = param_hash
        return

    os.makedirs(step_dir, exist_ok=True)

    summary_file = os.path.join(step_dir, f"summary_stats_{param_hash}.csv")
    metadata_file = os.path.join(step_dir, f"column_metadata_{param_hash}.json")
    class_plot_file = os.path.join(step_dir, f"class_distribution_{param_hash}.png")

    df.describe(include="all").to_csv(summary_file)

    metadata = {
        "param_hash": param_hash,
        "column_summary": {
            col: {
                "dtype": str(df[col].dtype),
                "cardinality": int(df[col].nunique(dropna=False)),
                "nulls": int(df[col].isnull().sum())
            } for col in df.columns
        }
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    sns.countplot(x=self.config["target_col"], data=df)
    plt.title("Class Distribution")
    plt.savefig(class_plot_file)
    plt.close()

    manifest = {
        "step": "eda",
        "param_hash": param_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "summary_stats_csv": summary_file,
            "column_metadata_json": metadata_file,
            "class_distribution_plot": class_plot_file
        }
    }
    with open(os.path.join(step_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step_key}_{param_hash}"):
            mlflow.set_tags({"step": step_key, "param_hash": param_hash})
            mlflow.log_params({"target_col": self.config["target_col"]})
            mlflow.log_artifacts(step_dir, artifact_path=step_key)

    log_registry(step=step_key, param_hash=param_hash, config=config, output_dir=step_dir)
    self.paths[step_key] = step_dir
    self.hashes[step_key] = param_hash

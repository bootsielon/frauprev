import os
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from utils import load_data, make_param_hash, log_registry

def eda(self) -> None:
    """
    Step 0: Perform EDA. Saves summary statistics, column metadata, and class distribution plot.
    Updates: self.dataframes["raw"], self.paths["eda"], self.hashes["eda"]
    """
    df = self.load_data()
    self.dataframes["raw"] = df

    config = {
        "target_column": self.config["target_column"],
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

    sns.countplot(x=self.config["target_column"], data=df)
    plt.title("Class Distribution")
    plt.savefig(class_plot_file)
    plt.close()

    manifest = {
        "step": step_key,
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
            mlflow.log_params({"target_column": self.config["target_column"]})
            mlflow.log_artifacts(step_dir, artifact_path=step_key)

    log_registry(step=step_key, param_hash=param_hash, config=config, output_dir=step_dir)
    self.paths[step_key] = step_dir
    self.hashes[step_key] = param_hash

# Helper function to mimic the original run_eda interface for testing.
def run_eda(df: pd.DataFrame, config: dict, param_hash: str, output_dir: str = "artifacts/eda", use_mlflow: bool = False) -> str:
    # Create a dummy pipeline object with the necessary attributes.
    class DummyPipeline:
        pass
    dp = DummyPipeline()
    dp.config = {"target_col": config["target_column"]}
    dp.dataframes = {}
    dp.paths = {}
    dp.hashes = {}
    dp.load_data = lambda db_path="fraud_poc.db": df
    eda(dp)
    return dp.paths.get("eda", output_dir)

if __name__ == "__main__":
    df = load_data()
    config = {
        "target_column": "is_fraud",
        "columns": sorted(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }
    param_hash = make_param_hash(config)
    run_eda(df, config=config, param_hash=param_hash, use_mlflow=True)
    print("EDA completed. Outputs saved to 'artifacts/eda' directory.")
    print(df.head())  # Display the first few rows of the DataFrame

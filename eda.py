import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime, timezone
from utils import load_data, make_param_hash  # Assuming these functions are in utils.py
import matplotlib
from matplotlib import font_manager as fm
import warnings
import mlflow
# Step 0: Modular, reproducible EDA with artifact hash and caching
# Prepare filesystem-safe utility for parameter hashing
import hashlib
import json
import os

def make_param_hash(params: dict) -> str:
    """
    Creates a deterministic hash from a dict of parameters to use in filenames.

    Args:
        params (dict): Dictionary of parameters used to generate outputs.

    Returns:
        str: A short alphanumeric hash string representing the parameters.
    """  # make_param_hash  # expose function for reuse
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:10]


# Debug available fonts (optional)
def debug_fonts():
    available_fonts = sorted(set(f.name for f in fm.fontManager.ttflist))
    # print("Available fonts:", available_fonts)

# Use Matplotlib's default sans-serif font
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX rendering for simplicity
matplotlib.rcParams['axes.unicode_minus'] = False  # Disable warnings for unsupported glyphs

# Suppress all font-related warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Debug fonts (optional)
# debug_fonts()

# Inspect the data for debugging (optional)
# def debug_data(df, column):
    # print(f"Unique values in {column}: {df[column].unique()}")


def generate_eda_metadata(df: pd.DataFrame) -> dict:
    """Generate per-column dtype, nulls, and cardinality."""
    metadata = {
        col: {
            "dtype": str(df[col].dtype),
            "cardinality": int(df[col].nunique(dropna=False)),
            "nulls": int(df[col].isnull().sum())
        } for col in df.columns
    }
    return metadata


# Update sanitize_labels to handle Text objects
def sanitize_labels(labels):
    """Remove unsupported glyphs from labels."""
    return [label.get_text().encode("ascii", "ignore").decode("ascii") for label in labels]


def save_class_distribution(df: pd.DataFrame, target_column: str, output_path: str) -> None:
    sns.countplot(x=target_column, data=df)
    plt.title("Class Distribution")
    plt.savefig(os.path.join(output_path, "class_distribution.png"))
    plt.close()


def save_numerical_histogram(df: pd.DataFrame, column: str, output_path: str) -> None:
    sns.histplot(df[column], bins=50, kde=True)
    plt.title(f"{column} Distribution")
    plt.savefig(os.path.join(output_path, f"{column}_distribution.png"))
    plt.close()


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes text columns in the DataFrame by removing unsupported glyphs
    and converts binary columns to integers.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Sanitized DataFrame.
    """
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: x.encode("ascii", "ignore").decode("ascii") if isinstance(x, str) else x)
    
    # Convert binary columns to integers
    for col in df.columns:
        if df[col].dtype == "object" and df[col].apply(lambda x: isinstance(x, bytes)).all():
            # print(f"Converting binary column '{col}' to integers.")  # Debugging
            df[col] = df[col].apply(lambda x: int.from_bytes(x, byteorder='little'))
    
    return df


def save_numerical_histograms(df: pd.DataFrame, target_col: str, output_dir: str):
    """
    Save histograms for numerical columns and class distribution plot.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Target column for class distribution.
        output_dir (str): Directory to save the plots.
    """
    sns.countplot(x=target_col, data=df)
    plt.title("Class Distribution")
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()

    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        if col == target_col:
            continue
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f"{col} Distribution")
        plt.savefig(os.path.join(output_dir, f"{col}_distribution.png"))
        plt.close()


def run_eda(
    df: pd.DataFrame,
    target_col: str,
    param_hash: str,
    config: dict,
    output_dir: str = "artifacts/eda",
    use_mlflow: bool = False
) -> str:
    """
    Run Exploratory Data Analysis and save outputs using hash-based versioning.

    Args:
        df: Input DataFrame
        target_col: Name of the target variable
        param_hash: Unique hash for this config
        config: Full configuration dictionary
        output_dir: Base output folder for artifacts
        use_mlflow: Whether to log outputs to MLflow

    Returns:
        Path to the output folder
    """
    step_dir = os.path.join(output_dir, f"eda_{param_hash}")
    summary_file = os.path.join(step_dir, f"summary_stats_{param_hash}.csv")
    metadata_file = os.path.join(step_dir, f"metadata_{param_hash}.json")
    class_plot_file = os.path.join(step_dir, f"class_distribution_{param_hash}.png")
    manifest_file = os.path.join(step_dir, "manifest.json")

    if os.path.exists(manifest_file) and os.path.exists(summary_file):
        print(f"[EDA] Skipping — cached at {step_dir}")
        return step_dir

    os.makedirs(step_dir, exist_ok=True)

    # Save summary statistics
    df.describe(include="all").to_csv(summary_file)

    # Save metadata
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

    # Save class distribution plot
    sns.countplot(x=target_col, data=df)
    plt.title("Class Distribution")
    plt.savefig(class_plot_file)
    plt.close()

    # Save manifest
    manifest = {
        "step": "eda",
        "param_hash": param_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "summary_stats": summary_file,
            "metadata": metadata_file,
            "class_distribution_plot": class_plot_file
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    # Log artifacts to MLflow
    if use_mlflow:
        with mlflow.start_run(run_name=f"EDA_{param_hash}") as run:
            mlflow.set_tags({"step": "eda", "hash": param_hash})
            mlflow.log_params({"target_col": target_col})
            mlflow.log_artifacts(step_dir, artifact_path="eda")

    return step_dir


if __name__ == "__main__":
    df = load_data()
    config = {
        "target_col": "is_fraud",
        "columns": sorted(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }
    param_hash = make_param_hash(config)
    run_eda(df, target_col=config["target_col"], param_hash=param_hash, config=config, use_mlflow=True)
    print("EDA completed. Outputs saved to 'artifacts/eda' directory.")
    print(df.head())  # Display the first few rows of the DataFrame
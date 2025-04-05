import pandas as pd
import os
import json
from datetime import datetime
import mlflow


def run_numeric_conversion(
    df: pd.DataFrame,
    target_col: str,
    param_hash: str,
    config: dict,
    output_dir: str = "artifacts/step3",
    use_mlflow: bool = False
) -> tuple[pd.DataFrame, str]:
    """
    Convert categorical features to numeric using cardinality rules and one-hot encoding.

    Args:
        df: Input DataFrame
        target_col: Name of the target variable
        param_hash: Hash generated from config
        config: Dictionary with conversion parameters (c1, c2, b1, c3)
        output_dir: Folder for saving artifacts
        use_mlflow: Whether to log to MLflow

    Returns:
        Tuple of transformed DataFrame and artifact output path
    """
    step_dir = os.path.join(output_dir, f"step3_{param_hash}")
    final_csv = os.path.join(step_dir, f"numeric_converted_{param_hash}.csv")
    binning_file = os.path.join(step_dir, f"category_binning_{param_hash}.json")
    mapping_file = os.path.join(step_dir, f"onehot_mapping_{param_hash}.json")
    drop_file = os.path.join(step_dir, f"dropped_features_{param_hash}.json")
    manifest_file = os.path.join(step_dir, "manifest.json")

    if os.path.exists(manifest_file) and os.path.exists(final_csv):
        print(f"[STEP 3] Skipping — cached at {step_dir}")
        return pd.read_csv(final_csv), step_dir

    os.makedirs(step_dir, exist_ok=True)

    df = df.copy()
    total_rows = len(df)
    binning_map = {}
    onehot_map = {}
    reverse_map = {}
    dropped_features = []
    id_like_features = []

    for col in df.select_dtypes(include="object").columns:
        if col == target_col:
            continue
        cardinality = df[col].nunique(dropna=False)

        is_id_like = total_rows / cardinality <= config["c3"]
        low_card = cardinality <= config["c1"]
        mid_card = cardinality <= int(config["c2"] * total_rows)
        high_card = not (low_card or mid_card)

        if is_id_like and config.get("id_like_exempt", True):
            id_like_features.append(col)
            continue

        if low_card:
            continue  # Keep as-is
        elif mid_card or (high_card and config["b1"]):
            top_categories = df[col].value_counts().nlargest(config["c1"]).index
            binning_map[col] = {
                "type": "top_k",
                "top_k": config["c1"],
                "retained": list(top_categories)
            }
            df[col] = df[col].apply(lambda x: x if x in top_categories else "Other")
        else:
            binning_map[col] = {
                "type": "dropped",
                "cardinality": cardinality
            }
            dropped_features.append(col)

    df.drop(columns=dropped_features + id_like_features, inplace=True)

    cat_cols = df.select_dtypes(include="object").columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    for orig_col in cat_cols:
        mapped_cols = [col for col in df_encoded.columns if col.startswith(f"{orig_col}_")]
        onehot_map[orig_col] = mapped_cols
        for new_col in mapped_cols:
            reverse_map[new_col] = orig_col

    df_encoded.to_csv(final_csv, index=False)

    with open(binning_file, "w") as f:
        json.dump({"param_hash": param_hash, "config": config, "binning_map": binning_map}, f, indent=2)
    with open(mapping_file, "w") as f:
        json.dump({"param_hash": param_hash, "mapping": onehot_map, "reverse_mapping": reverse_map}, f, indent=2)
    with open(drop_file, "w") as f:
        json.dump({"param_hash": param_hash, "dropped_features": dropped_features, "id_like": id_like_features}, f, indent=2)

    manifest = {
        "step": "numeric_conversion",
        "param_hash": param_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "csv": final_csv,
            "binning": binning_file,
            "onehot_mapping": mapping_file,
            "drop_summary": drop_file
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    if use_mlflow:
        with mlflow.start_run(run_name=f"Step3_Numeric_{param_hash}"):
            mlflow.set_tags({"step": "numeric_conversion", "hash": param_hash})
            mlflow.log_params(config)
            mlflow.log_artifacts(step_dir, artifact_path="numeric_conversion")

    return df_encoded, step_dir


if __name__ == "__main__":
    from utils import load_data, make_param_hash

    df_raw = load_data()
    config = {
        "target_col": "is_fraud",
        "c1": 10,
        "c2": 0.01,
        "b1": True,
        "c3": 10,
        "id_like_exempt": True
    }
    hash_id = make_param_hash(config)
    df_num, out_path = run_numeric_conversion(df_raw, target_col=config["target_col"], param_hash=hash_id, config=config, use_mlflow=True)
    print(f"[TEST] Numeric features saved to: {out_path}")

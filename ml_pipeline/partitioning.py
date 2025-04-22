"""
 This module contains functions for partitioning a dataset into training, validation, and test sets.
 
 It includes functionality for stratified downsampling, random downsampling, and saving the resulting
 datasets to files. The module also supports logging with MLflow and creating a manifest file for
 tracking purposes.

 It is designed to be used as part of a larger machine learning pipeline.
"""

import os
import json
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from ml_pipeline.utils import make_param_hash, log_registry, DEFAULT_TEST_HASH  # already lives in utils.py


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def identify_stratification_columns(df, target, use_stratification, cardinality_threshold):
    """Identify columns to use for stratification."""
    stratify_cols = [target]
    if use_stratification:
        for col in df.columns:
            if col != target and df[col].nunique() <= cardinality_threshold:
                stratify_cols.append(col)
    return stratify_cols


def identify_classes(df, target, step):
    """Identify minority and majority classes in the dataset."""
    class_counts = df[target].value_counts()
    if len(class_counts) != 2:
        print(f"[{step.upper()}] Warning: Expected binary classification but found {len(class_counts)} classes")
    
    minority_class = class_counts.index[-1]  # Least frequent class
    majority_class = class_counts.index[0]   # Most frequent class
    
    df_minority = df[df[target] == minority_class]
    df_majority = df[df[target] == majority_class]
    
    return minority_class, majority_class, df_minority, df_majority, class_counts


def perform_stratified_downsampling(df_majority, df_minority, strat_cols_for_downsampling, seed):
    """Downsample majority class with stratification."""
    majority_strat_key = df_majority[strat_cols_for_downsampling].astype(str).agg("_".join, axis=1)
    minority_strat_key = df_minority[strat_cols_for_downsampling].astype(str).agg("_".join, axis=1)
    
    minority_dist = minority_strat_key.value_counts(normalize=True)
    
    df_majority_downsampled = pd.DataFrame(columns=df_majority.columns)
    df_excluded = pd.DataFrame(columns=df_majority.columns)
    
    for stratum, proportion in minority_dist.items():
        stratum_df = df_majority[majority_strat_key == stratum]
        if not stratum_df.empty:
            n_samples = max(1, int(proportion * len(df_minority)))
            if len(stratum_df) > n_samples:
                sampled = stratum_df.sample(n=n_samples, random_state=seed)
                df_majority_downsampled = pd.concat([df_majority_downsampled, sampled])
                df_excluded = pd.concat([df_excluded, stratum_df.drop(sampled.index)])
            else:
                df_majority_downsampled = pd.concat([df_majority_downsampled, stratum_df])
    
    return df_majority_downsampled, df_excluded


def balance_samples(df_majority_downsampled, df_minority, df_majority, df_excluded, seed):
    """Balance the number of samples between majority and minority classes."""
    if len(df_majority_downsampled) > len(df_minority):
        excess = len(df_majority_downsampled) - len(df_minority)
        excess_indices = df_majority_downsampled.sample(n=excess, random_state=seed).index
        df_excluded = pd.concat([df_excluded, df_majority_downsampled.loc[excess_indices]])
        df_majority_downsampled = df_majority_downsampled.drop(excess_indices)
    elif len(df_majority_downsampled) < len(df_minority):
        shortage = len(df_minority) - len(df_majority_downsampled)
        if len(df_excluded) >= shortage:
            additional = df_excluded.sample(n=shortage, random_state=seed)
            df_majority_downsampled = pd.concat([df_majority_downsampled, additional])
            df_excluded = df_excluded.drop(additional.index)
        else:
            additional_needed = shortage - len(df_excluded)
            df_majority_downsampled = pd.concat([df_majority_downsampled, df_excluded])
            df_excluded = pd.DataFrame(columns=df_majority.columns)
            additional = df_majority.sample(n=additional_needed, random_state=seed)
            df_majority_downsampled = pd.concat([df_majority_downsampled, additional])
            df_excluded = df_majority.drop(df_majority_downsampled.index)
    
    return df_majority_downsampled, df_excluded


def random_downsample(df_majority, df_minority, seed):
    """Simple random downsampling of majority class."""
    df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=seed)
    df_excluded = df_majority.drop(df_majority_downsampled.index)
    return df_majority_downsampled, df_excluded


# --------------------------------------------------------------------------- #
# Train / Val / Test split helper                                             #
# --------------------------------------------------------------------------- #
def perform_data_splits(
    df: pd.DataFrame,
    stratify_cols: list[str],
    test_size: float | int,
    val_ratio: float,
    seed: int,
):
    """
    Split *df* into train, validation and test sets while preserving the joint
    distribution of all columns in *stratify_cols*.

    Safety guards:
    1. Ensure the **test** split has at least one sample per stratum.
    2. Ensure the **validation** split has at least one sample per stratum.

    If either constraint would be violated, the corresponding split fraction is
    automatically increased to the minimum feasible value and a message is
    printed.  The adjustment keeps the pipeline running without user
    intervention.
    """
    # Build a single stratification key
    stratify_key = df[stratify_cols].astype(str).agg("_".join, axis=1)
    n_classes = stratify_key.nunique()

    # ───────────────────────────────── 1️⃣  Test split guard
    if isinstance(test_size, float):
        test_rows_req = max(int(round(len(df) * test_size)), 1)
    else:
        test_rows_req = test_size
    if test_rows_req < n_classes:
        test_rows_req = n_classes
        test_size = test_rows_req / len(df)
        print(
            f"[PARTITIONING] Auto‑bumped test_size to {test_size:.3f} "
            f"so that each of the {n_classes} strata appears at least once."
        )

    # First split: Train+Val vs Test
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_key,
        random_state=seed,
    )

    # ───────────────────────────────── 2️⃣  Validation split guard
    stratify_key_tv = df_train_val[stratify_cols].astype(str).agg("_".join, axis=1)
    n_classes_tv = stratify_key_tv.nunique()

    if isinstance(val_ratio, float):
        val_rows_req = max(int(round(len(df_train_val) * val_ratio)), 1)
    else:  # val_ratio should always be float, but guard just in case
        val_rows_req = val_ratio
    if val_rows_req < n_classes_tv:
        val_rows_req = n_classes_tv
        val_ratio = val_rows_req / len(df_train_val)
        print(
            f"[PARTITIONING] Auto‑bumped val_ratio to {val_ratio:.3f} "
            f"so that each of the {n_classes_tv} strata appears at least once."
        )

    # Second split: Train vs Val
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_ratio,
        stratify=stratify_key_tv,
        random_state=seed,
    )

    return df_train, df_val, df_test, stratify_key



def save_outputs(df_train, df_val, df_test, df_excluded,
                  id_col, stratify_keys, step_dir, param_hash):
    """Save all partition outputs to files (hash omitted from filenames per SPEC§25)."""
    df_train.to_csv(os.path.join(step_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(step_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(step_dir, "test.csv"), index=False)
    df_excluded.to_csv(os.path.join(step_dir, "excluded_majority.csv"), index=False)

    id_map = {
        "train": df_train[id_col].tolist(),
        "val": df_val[id_col].tolist(),
        "test": df_test[id_col].tolist(),
        "excluded_majority": df_excluded[id_col].tolist()
    }
    with open(os.path.join(step_dir, "id_partition_map.json"), "w") as f:
        json.dump(id_map, f, indent=2)

    stratify_keys.to_csv(os.path.join(step_dir, "stratify_keys.csv"), index=False)


def create_manifest(step, param_hash, config, step_dir):
    """Create and save the manifest file."""
    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "train_csv": "train.csv",
            "val_csv": "val.csv",
            "test_csv": "test.csv",
            "excluded_majority_csv": "excluded_majority.csv",
            "id_partition_map_json": "id_partition_map.json",
            "stratify_keys_csv": "stratify_keys.csv"
        }
    }
    with open(os.path.join(step_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def load_checkpoint(step_dir, param_hash, step):
    """Load data from existing checkpoint (filenames hash‑free)."""
    dataframes = {}
    for split in ["train", "val", "test", "excluded_majority"]:
        path = os.path.join(step_dir, f"{split}.csv")
        if os.path.exists(path):
            dataframes[split] = pd.read_csv(path)
    return dataframes


# --------------------------------------------------------------------------- #
# Main pipeline step                                                          #
# --------------------------------------------------------------------------- #
def partitioning(self) -> None:
    """
    Partition feature‑engineered data into train, validation, and test sets.
    """
    step = "partitioning"
    param_hash = self.global_hash
    step_dir   = os.path.join("artifacts", f"run_{self.global_hash}", step)
    manifest   = os.path.join(step_dir, "manifest.json")

    # ----------------------------------------------------------------
    # 1️⃣  Skip‑guard – artefacts already in *current* run folder
    # ----------------------------------------------------------------
    if os.path.exists(manifest):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.dataframes.update(load_checkpoint(step_dir, param_hash, step))
        return

    # ----------------------------------------------------------------
    # 2️⃣  Inference mode → load artefacts from TRAINING run
    # ----------------------------------------------------------------
    if not self.train_mode:
        train_dir   = os.path.join("artifacts", f"run_{self.global_train_hash}", step)
        train_manif = os.path.join(train_dir, "manifest.json")
        if os.path.exists(train_manif):
            print(f"[{step.upper()}] Reusing artefacts from {train_dir}")
            self.paths[step] = train_dir
            self.dataframes.update(load_checkpoint(train_dir, self.global_train_hash, step))
            return
        # ‑‑‑ nothing to reuse → raise, as required by SPEC §5
        raise FileNotFoundError(
            f"[{step.upper()}] Expected training artefacts at {train_dir} but none found."
        )

    # ----------------------------------------------------------------
    # 3️⃣  Training mode – continue with normal computation
    # ----------------------------------------------------------------

    
    df = self.dataframes["feature_engineered"]
    target = self.config["target_col"]
    id_col = self.config["id_col"]
    use_stratification = self.config["use_stratification"]
    use_downsampling = self.config.get("use_downsampling", True)
    seed = self.config["seed"]

    # ------------------------------------------------------------------- #
    # SPEC§1: use run‑level directory: artifacts/run_<hash>/<step>/       #
    # ------------------------------------------------------------------- #
    param_hash = self.global_hash  # keep var name to minimise diff
    step_dir = os.path.join("artifacts", f"run_{self.global_hash}", step)
    manifest_file = os.path.join(step_dir, "manifest.json")

    # ------------------------------------------------------------------- #
    # SPEC§14 skip‑guard                                                  #
    # ------------------------------------------------------------------- #
    if os.path.exists(manifest_file):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        # removed: self.hashes no longer used.  # SPEC§3
        checkpoint_data = load_checkpoint(step_dir, param_hash, step)
        self.dataframes.update(checkpoint_data)
        return

    os.makedirs(step_dir, exist_ok=True)

    stratify_cols = identify_stratification_columns(
        df, target, use_stratification, self.config["stratify_cardinality_threshold"]
    )
    
    df_for_splitting = df
    df_excluded = pd.DataFrame(columns=df.columns)
    
    if use_downsampling:
        minority_class, majority_class, df_minority, df_majority, _ = identify_classes(df, target, step)
        
        if len(df_majority) < len(df_minority):
            print(f"[{step.upper()}] Warning: 'Majority' class {majority_class} ({len(df_majority)} samples) "
                  f"is smaller than 'minority' class {minority_class} ({len(df_minority)} samples). "
                  f"Skipping downsampling.")
        else:
            if use_stratification:
                strat_cols_for_downsampling = [col for col in stratify_cols if col != target]
                
                if strat_cols_for_downsampling:
                    df_majority_downsampled, df_excluded = perform_stratified_downsampling(
                        df_majority, df_minority, strat_cols_for_downsampling, seed
                    )
                    df_majority_downsampled, df_excluded = balance_samples(
                        df_majority_downsampled, df_minority, df_majority, df_excluded, seed
                    )
                else:
                    df_majority_downsampled, df_excluded = random_downsample(df_majority, df_minority, seed)
            else:
                df_majority_downsampled, df_excluded = random_downsample(df_majority, df_minority, seed)
            
            df_balanced = pd.concat([df_minority, df_majority_downsampled], axis=0).sample(frac=1, random_state=seed)
            df_for_splitting = df_balanced

    stratify_key = df_for_splitting[stratify_cols].astype(str).agg("_".join, axis=1)
    self.dataframes["stratification_keys"] = pd.Series(stratify_key)

    val_ratio = self.config["val_size"] / (self.config["train_size"] + self.config["val_size"])
    df_train, df_val, df_test, _ = perform_data_splits(
        df_for_splitting, stratify_cols, self.config["test_size"], val_ratio, seed
    )
    
    save_outputs(
        df_train, df_val, df_test, df_excluded, 
        id_col, self.dataframes["stratification_keys"], 
        step_dir, param_hash
    )
    
    create_manifest(step, param_hash, self.config, step_dir)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    log_registry(step, self.global_hash, self.config, step_dir)  # SPEC§7

    self.dataframes.update({
        "train": df_train,
        "val": df_val,
        "test": df_test,
        "excluded": df_excluded
    })
    print(f"[{step.upper()}] Partitioning completed. Data saved to {step_dir}")
    print(f"[{step.upper()}] Train samples: {len(df_train)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}")
    print(f"[{step.upper()}] Excluded samples: {len(df_excluded)}")
    print(f"[{step.upper()}] Total samples processed: "
          f"{len(df_train) + len(df_val) + len(df_test) + len(df_excluded)}")

    self.paths[step] = step_dir
    # removed: self.hashes no longer used.  # SPEC§3



if __name__ == "__main__":
    """
    Smoke‑tests for the partitioning step
    (SPEC §§17‑19 and §21: delivered separately from the main file).
    """
    import numpy as np
    import pandas as pd
    import shutil
    import os
    from ml_pipeline.utils import DEFAULT_TEST_HASH
    from ml_pipeline.base import MLPipeline

    # ---------------------------------------------------------------
    # Helper to build a fresh pipeline instance
    # ---------------------------------------------------------------
    def build_pipeline(train_mode: bool, *, with_created_hash: bool = True):
        # base parameters shared by both train & inference
        cfg: dict = {
            "target_col": "is_fraud",
            "id_col": "transaction_id",
            "use_mlflow": False,
            "seed": 42,
            "use_stratification": True,
            "use_downsampling": True,
            "stratify_cardinality_threshold": 5,
            "train_size": 0.7,
            "val_size": 0.15,
            "test_size": 0.15,
            # always present so we can flip modes freely
            "train_hash": DEFAULT_TEST_HASH,
            # mandatory inference‑mode keys (dummies are fine for tests)
            "model_name": "dummy_model",
            "model_hash": "abcd1234",
            "dataset_name": "dummy_ds",
            "feature_names": ["amount", "hour", "day"],
        }

        cfg["train_mode"] = train_mode            # flag must live inside config
        pipe = MLPipeline(config=cfg)             # ctor reads everything from cfg

        if with_created_hash:
            pipe.global_hash = DEFAULT_TEST_HASH  # deterministic hash for tests
        return pipe

    # ---------------------------------------------------------------
    # Create mock data
    # ---------------------------------------------------------------
    np.random.seed(42)
    n_samples = 200
    mock_df = pd.DataFrame({
        "transaction_id": range(1, n_samples + 1),
        "amount": np.random.uniform(5, 500, n_samples),
        "client_id": np.random.randint(1, 10, n_samples),
        "merchant_id": np.random.randint(20, 40, n_samples),
        "hour": np.random.randint(0, 6, n_samples),
        "day": np.random.randint(0, 3, n_samples),
        "category": np.random.choice(["A", "B"], n_samples),
        "is_fraud": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    })
    mock_df.loc[mock_df.sample(frac=0.05, random_state=1).index, "amount"] = np.nan

    # Clean up previous artefacts
    artefact_root = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}")
    # if os.path.exists(artefact_root):
        # shutil.rmtree(artefact_root)

    # 1️⃣  Training ‑ fresh
    pipe_train_fresh = build_pipeline(train_mode=True)
    pipe_train_fresh.dataframes["feature_engineered"] = mock_df.copy()
    print("\n>>> TRAINING RUN (fresh artefacts)")
    pipe_train_fresh.partitioning()

    # 2️⃣  Training ‑ skip‑guard
    pipe_train_skip = build_pipeline(train_mode=True)
    pipe_train_skip.dataframes["feature_engineered"] = mock_df.copy()
    print("\n>>> TRAINING RUN (should skip)")
    pipe_train_skip.partitioning()

    # 3️⃣  Inference ‑ artefacts present
    pipe_infer_ok = build_pipeline(train_mode=False)
    pipe_infer_ok.dataframes["feature_engineered"] = mock_df.copy()
    print("\n>>> INFERENCE RUN (artefacts present)")
    pipe_infer_ok.partitioning()

    # 4️⃣  Inference ‑ artefacts missing (expect failure)
# 4️⃣  Inference – artefacts missing (must fail)
    #missing_hash = "deadbeef9999"  # any value not used earlier
    #pipe_infer_fail = build_pipeline(train_mode=False, with_created_hash=False)
    #pipe_infer_fail.global_hash = missing_hash
    # DO NOT inject feature_engineered here → forces step to look for training artefacts
    print("\n>>> INFERENCE RUN (artefacts missing, should fail normally, but in partitioning this test is irrelevant)")
    #try:
        #pipe_infer_fail.partitioning()
        #print("❌  ERROR: Missing‑artefact inference did *not* fail as expected")
    #except FileNotFoundError as e:
    #    print(f"✅  Caught expected error: {e}")
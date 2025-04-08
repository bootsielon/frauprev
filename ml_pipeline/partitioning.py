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
from .utils import make_param_hash
from .utils import log_registry


def identify_stratification_columns(df, target, use_stratification, cardinality_threshold):
    """Identify columns to use for stratification.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing features.
        target (str): Name of the target column.
        use_stratification (bool): Whether to use stratification based on categorical features.
        cardinality_threshold (int): Maximum number of unique values a column can have to be used for stratification.
    
    Returns:
        list: List of column names to use for stratification.
    """
    stratify_cols = [target]
    if use_stratification:
        for col in df.columns:
            if col != target and df[col].nunique() <= cardinality_threshold:
                stratify_cols.append(col)
    return stratify_cols


def identify_classes(df, target, step):
    """Identify minority and majority classes in the dataset.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target (str): Name of the target column.
        step (str): Name of the pipeline step for logging purposes.
    
    Returns:
        tuple: A tuple containing:
            - minority_class: The value representing the minority class
            - majority_class: The value representing the majority class
            - df_minority: DataFrame containing only minority class samples
            - df_majority: DataFrame containing only majority class samples
            - class_counts: Series containing class distribution counts
    """
    class_counts = df[target].value_counts()
    if len(class_counts) != 2:
        print(f"[{step.upper()}] Warning: Expected binary classification but found {len(class_counts)} classes")
    
    # Get classes sorted by frequency
    minority_class = class_counts.index[-1]  # Least frequent class
    majority_class = class_counts.index[0]   # Most frequent class
    
    df_minority = df[df[target] == minority_class]
    df_majority = df[df[target] == majority_class]
    
    return minority_class, majority_class, df_minority, df_majority, class_counts


def perform_stratified_downsampling(df_majority, df_minority, strat_cols_for_downsampling, seed):
    """Downsample majority class with stratification.
    
    This function performs stratified downsampling of the majority class to match
    the distribution of categorical variables in the minority class.
    
    Args:
        df_majority (pd.DataFrame): DataFrame containing majority class samples.
        df_minority (pd.DataFrame): DataFrame containing minority class samples.
        strat_cols_for_downsampling (list): List of column names to use for stratification.
        seed (int): Random seed for reproducibility.
    
    Returns:
        tuple: A tuple containing:
            - df_majority_downsampled: DataFrame containing downsampled majority class
            - df_excluded: DataFrame containing excluded majority class samples
    """
    majority_strat_key = df_majority[strat_cols_for_downsampling].astype(str).agg("_".join, axis=1)
    minority_strat_key = df_minority[strat_cols_for_downsampling].astype(str).agg("_".join, axis=1)
    
    # Count distribution in minority class
    minority_dist = minority_strat_key.value_counts(normalize=True)
    
    # Initialize empty dataframe for stratified downsampling
    df_majority_downsampled = pd.DataFrame(columns=df_majority.columns)
    df_excluded = pd.DataFrame(columns=df_majority.columns)
    
    # Sample from each stratum
    for stratum, proportion in minority_dist.items():
        stratum_df = df_majority[majority_strat_key == stratum]
        if not stratum_df.empty:
            # Calculate how many samples to take from this stratum
            n_samples = max(1, int(proportion * len(df_minority)))
            if len(stratum_df) > n_samples:
                sampled = stratum_df.sample(n=n_samples, random_state=seed)
                df_majority_downsampled = pd.concat([df_majority_downsampled, sampled])
                df_excluded = pd.concat([df_excluded, stratum_df.drop(sampled.index)])
            else:
                # If not enough samples in this stratum, take all
                df_majority_downsampled = pd.concat([df_majority_downsampled, stratum_df])
    
    return df_majority_downsampled, df_excluded


def balance_samples(df_majority_downsampled, df_minority, df_majority, df_excluded, seed):
    """Balance the number of samples between majority and minority classes.

    This function ensures that the number of samples in the downsampled majority class
    matches the number of samples in the minority class. It also handles cases where
    there are not enough excluded samples to balance the classes.

    Args:
        df_majority_downsampled (pd.DataFrame): Downsampled majority class samples.
        df_minority (pd.DataFrame): DataFrame containing minority class samples.
        df_majority (pd.DataFrame): DataFrame containing original majority class samples.
        df_excluded (pd.DataFrame): DataFrame containing excluded samples from downsampling.
        seed (int): Random seed for reproducibility.    
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Downsampled majority class and excluded samples.
    """
    # Ensure we have exactly the same number of majority samples as minority
    if len(df_majority_downsampled) > len(df_minority):
        # Randomly remove excess samples
        excess = len(df_majority_downsampled) - len(df_minority)
        excess_indices = df_majority_downsampled.sample(n=excess, random_state=seed).index
        df_excluded = pd.concat([df_excluded, df_majority_downsampled.loc[excess_indices]])
        df_majority_downsampled = df_majority_downsampled.drop(excess_indices)
    elif len(df_majority_downsampled) < len(df_minority):
        # Randomly add more samples if needed
        shortage = len(df_minority) - len(df_majority_downsampled)
        if len(df_excluded) >= shortage:
            additional = df_excluded.sample(n=shortage, random_state=seed)
            df_majority_downsampled = pd.concat([df_majority_downsampled, additional])
            df_excluded = df_excluded.drop(additional.index)
        else:
            # If we don't have enough excluded samples, fall back to random sampling
            additional_needed = shortage - len(df_excluded)
            df_majority_downsampled = pd.concat([df_majority_downsampled, df_excluded])
            df_excluded = pd.DataFrame(columns=df_majority.columns)
            # Random sample from original majority to make up the difference
            additional = df_majority.sample(n=additional_needed, random_state=seed)
            df_majority_downsampled = pd.concat([df_majority_downsampled, additional])
            df_excluded = df_majority.drop(df_majority_downsampled.index)
    
    return df_majority_downsampled, df_excluded


def random_downsample(df_majority, df_minority, seed):
    """Simple random downsampling of majority class.
    
    This function randomly samples the majority class to match the size of the minority class.
    It is used when stratified downsampling is not applicable or desired.

    Args:
        df_majority (pd.DataFrame): DataFrame containing majority class samples.
        df_minority (pd.DataFrame): DataFrame containing minority class samples.
        seed (int): Random seed for reproducibility.

        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Downsampled majority class and excluded samples.
    
    """
    df_majority_downsampled = df_majority.sample(
        n=len(df_minority),
        random_state=seed
    )
    df_excluded = df_majority.drop(df_majority_downsampled.index)
    return df_majority_downsampled, df_excluded


def perform_data_splits(df, stratify_cols, test_size, val_ratio, seed):
    """Split data into train, validation and test sets.
    
    This function splits the input DataFrame into training, validation, and test sets
    while preserving the stratification based on the specified columns.

    Args:
        df (pd.DataFrame): DataFrame to be split.
        stratify_cols (list): List of columns to use for stratification.
        test_size (float): Proportion of the dataset to include in the test split.
        val_ratio (float): Proportion of the training data to include in the validation split.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]: Train, validation, and test DataFrames,
        and the stratification key Series.
    """
    stratify_key = df[stratify_cols].astype(str).agg("_".join, axis=1)
    
    # Train/Val/Test Split
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_key,
        random_state=seed
    )
    
    stratify_key_tv = df_train_val[stratify_cols].astype(str).agg("_".join, axis=1)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_ratio,
        stratify=stratify_key_tv,
        random_state=seed
    )
    
    return df_train, df_val, df_test, stratify_key


def save_outputs(df_train, df_val, df_test, df_excluded,
                  id_col, stratify_keys, step_dir, param_hash):
    """
    Save all partition outputs to files.

    This function saves the training, validation, test, and excluded samples DataFrames
    to CSV files. It also creates a JSON file containing the partition mapping for each split.
    Additionally, it saves the stratification keys used for the splits.

    Args:
        df_train (pd.DataFrame): Training DataFrame.
        df_val (pd.DataFrame): Validation DataFrame.
        df_test (pd.DataFrame): Test DataFrame.
        df_excluded (pd.DataFrame): Excluded samples DataFrame.
        id_col (str): Column name for unique identifiers.
        stratify_keys (pd.Series): Series containing the stratification keys.
        step_dir (str): Directory to save the outputs.
        param_hash (str): Hash of the parameters used for this step.
    """
    # Save datasets
    df_train.to_csv(os.path.join(step_dir, f"train_{param_hash}.csv"), index=False)
    df_val.to_csv(os.path.join(step_dir, f"val_{param_hash}.csv"), index=False)
    df_test.to_csv(os.path.join(step_dir, f"test_{param_hash}.csv"), index=False)
    df_excluded.to_csv(os.path.join(step_dir, f"excluded_majority_{param_hash}.csv"), index=False)

    # Save partition map
    id_map = {
        "train": df_train[id_col].tolist(),
        "val": df_val[id_col].tolist(),
        "test": df_test[id_col].tolist(),
        "excluded_majority": df_excluded[id_col].tolist()
    }
    with open(os.path.join(step_dir, f"id_partition_map_{param_hash}.json"), "w") as f:
        json.dump(id_map, f, indent=2)

    # Save stratification key info
    stratify_csv = os.path.join(step_dir, f"stratify_keys_{param_hash}.csv")
    stratify_keys.to_csv(stratify_csv, index=False)


def create_manifest(step, param_hash, config, step_dir):
    """Create and save the manifest file.

    This function creates a manifest file that contains metadata about the partitioning step,
    including the step name, parameter hash, timestamp, configuration, output directory,
    and output files. The manifest is saved as a JSON file in the specified directory.

    Args:
        step (str): Name of the step (e.g., "partitioning").
        param_hash (str): Hash of the parameters used for this step.
        config (dict): Configuration dictionary containing pipeline settings.
        step_dir (str): Directory to save the manifest.
    """
    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "train_csv": f"train_{param_hash}.csv",
            "val_csv": f"val_{param_hash}.csv",
            "test_csv": f"test_{param_hash}.csv",
            "excluded_majority_csv": f"excluded_majority_{param_hash}.csv",
            "id_partition_map_json": f"id_partition_map_{param_hash}.json",
            "stratify_keys_csv": f"stratify_keys_{param_hash}.csv"
        }
    }
    manifest_file = os.path.join(step_dir, "manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)


def load_checkpoint(step_dir, param_hash, step):
    """Load data from existing checkpoint.
    
    This function loads previously saved DataFrames from a checkpoint directory.
    
    Args:
        step_dir (str): Directory containing the checkpoint files.
        param_hash (str): Hash of the parameters used for this step.
        step (str): Name of the pipeline step.
    
    Returns:
        dict: Dictionary containing DataFrames for each split (train, val, test, excluded_majority).
    """
    dataframes = {}
    for split in ["train", "val", "test", "excluded_majority"]:
        path = os.path.join(step_dir, f"{split}_{param_hash}.csv")
        if os.path.exists(path):
            dataframes[split] = pd.read_csv(path)
    return dataframes


def partitioning(self) -> None:
    """Partition feature-engineered data into train, validation, and test sets.
    
    This function performs several key operations:
    1. Checks for existing checkpoints to avoid redundant processing
    2. Identifies columns for stratification based on configuration
    3. Applies downsampling to balance classes if configured
    4. Splits data into train, validation, and test sets
    5. Saves outputs and records metadata
    6. Updates pipeline state with new DataFrames and paths
    
    The function supports both random and stratified downsampling, as well as
    different splitting strategies based on configuration parameters.
    
    Updates:
        self.dataframes: Updates with "train", "val", "test", "excluded_majority", and "stratification_keys"
        self.paths: Updates with the path to the partitioning outputs
        self.hashes: Updates with the hash for the partitioning step
    """
    step = "partitioning"
    df = self.dataframes["feature_engineered"]
    target = self.config["target_col"]
    id_col = self.config["id_col"]
    use_stratification = self.config["use_stratification"]
    use_downsampling = self.config.get("use_downsampling", True)
    seed = self.config["seed"]

    param_hash = make_param_hash(self.config)
    step_dir = os.path.join("artifacts", f"{step}_{param_hash}")
    manifest_file = os.path.join(step_dir, "manifest.json")

    # Check for checkpoint
    if os.path.exists(manifest_file):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = param_hash
        checkpoint_data = load_checkpoint(step_dir, param_hash, step)
        self.dataframes.update(checkpoint_data)
        return

    os.makedirs(step_dir, exist_ok=True)

    # Identify stratification columns
    stratify_cols = identify_stratification_columns(
        df, target, use_stratification, self.config["stratify_cardinality_threshold"]
    )
    
    # Initialize default
    df_for_splitting = df
    df_excluded = pd.DataFrame(columns=df.columns)
    
    # Apply downsampling if configured
    if use_downsampling:
        # Identify minority and majority classes
        minority_class, majority_class, df_minority, df_majority, class_counts = identify_classes(df, target, step)
        
        # Skip downsampling if minority class is actually larger
        if len(df_majority) < len(df_minority):
            print(f"[{step.upper()}] Warning: 'Majority' class {majority_class} ({len(df_majority)} samples) is smaller than "
                  f"'minority' class {minority_class} ({len(df_minority)} samples). Skipping downsampling.")
        else:
            # Apply appropriate downsampling strategy
            if use_stratification:
                # Stratified downsampling - exclude target from stratification columns
                strat_cols_for_downsampling = [col for col in stratify_cols if col != target]
                
                if strat_cols_for_downsampling:
                    # Perform stratified downsampling
                    df_majority_downsampled, df_excluded = perform_stratified_downsampling(
                        df_majority, df_minority, strat_cols_for_downsampling, seed
                    )
                    
                    # Balance sample counts if needed
                    df_majority_downsampled, df_excluded = balance_samples(
                        df_majority_downsampled, df_minority, df_majority, df_excluded, seed
                    )
                else:
                    # Fall back to random downsampling if no stratification columns remain
                    df_majority_downsampled, df_excluded = random_downsample(df_majority, df_minority, seed)
            else:
                # Random downsampling
                df_majority_downsampled, df_excluded = random_downsample(df_majority, df_minority, seed)
            
            # Create balanced dataset for splitting
            df_balanced = pd.concat([df_minority, df_majority_downsampled], axis=0).sample(frac=1, random_state=seed)
            df_for_splitting = df_balanced

    # Create stratification key for splits
    stratify_key = df_for_splitting[stratify_cols].astype(str).agg("_".join, axis=1)
    self.dataframes["stratification_keys"] = pd.Series(stratify_key)

    # Split data into train, validation and test sets
    val_ratio = self.config["val_size"] / (self.config["train_size"] + self.config["val_size"])
    df_train, df_val, df_test, _ = perform_data_splits(
        df_for_splitting, stratify_cols, self.config["test_size"], val_ratio, seed
    )
    
    # Save all outputs
    save_outputs(
        df_train, df_val, df_test, df_excluded, 
        id_col, self.dataframes["stratification_keys"], 
        step_dir, param_hash
    )
    
    # Create and save manifest
    create_manifest(step, param_hash, self.config, step_dir)

    # Log with MLflow if configured
    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    # Log to registry
    log_registry(step, param_hash, self.config, step_dir)

    # Store in pipeline state
    self.dataframes.update({
        "train": df_train,
        "val": df_val,
        "test": df_test,
        "excluded": df_excluded
    })
    print(f"[{step.upper()}] Partitioning completed. Data saved to {step_dir}")
    print(f"[{step.upper()}] Train samples: {len(df_train)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}")
    print(f"[{step.upper()}] Excluded samples: {len(df_excluded)}")
    print(f"[{step.upper()}] Total samples processed: {len(df_train) + len(df_val) + len(df_test) + len(df_excluded)}")

    self.paths[step] = step_dir
    self.hashes[step] = param_hash

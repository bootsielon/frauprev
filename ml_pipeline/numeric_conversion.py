import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import mlflow
from ml_pipeline.utils import make_param_hash, log_registry, convert_numpy_types


def numeric_conversion(self) -> None:
    """
    Step 3: Convert non-numeric variables to numeric representations using cardinality-based rules.
    Includes one-hot encoding and full mapping traceability.

    Updates:
        self.dataframes["train_num"], ["val_num"], ["test_num"]
        self.paths["numeric_conversion"]
        self.hashes["numeric_conversion"]
    """
    train_mode = self.config["train_mode"]

    step = "numeric_conversion"
    test_df = self.dataframes["test"]
    if train_mode:
        train_df = self.dataframes["train"]
        val_df = self.dataframes["val"]        
        excluded_df = self.dataframes.get("excluded", None)
        dataset_size = len(train_df)  # + (len(excluded_numeric) if excluded_numeric is not None else 0)
    else:
        dataset_size = len(test_df)  # Adjust dataset size for test mode

    config = {
        "c1": self.config.get("c1"),  # cardinality threshold for one-hot
        "c2": self.config.get("c2"),  # fraction threshold for rare category reduction
        "b1": self.config.get("b1"),  # treat high-cardinality vars as mid if True
        "c3": self.config.get("c3"),  # log-scale threshold for ID-like
        "id_like_exempt": self.config.get("id_like_exempt", True)
    }

    # param_hash = make_param_hash(config)  # if train_mode else self.config["inf_hash"]
    # if not train_mode: 
        # inf_hash = self.config["inf_hash"]
    # Generate a unique hash for the parameters used in this step
    # step_dir = os.path.join("artifacts", f"{step}_{param_hash}")

    if train_mode:
        # Training mode: generate hash from config
        param_hash = make_param_hash(config)
        step_dir = os.path.join("artifacts", f"{step}_{param_hash}")
    else:
        # Inference mode: 
        # 1. Use train_hash to load training artifacts
        train_hash = self.config["train_hash"]
        train_step_dir = os.path.join("artifacts", f"{step}_{train_hash}")
        
        # 2. Generate new hash for storing inference results (optional)
        inference_config = {
            "inference_time": datetime.now().isoformat(),
            "source_model_hash": train_hash,
            "test_data_shape": test_df.shape
        }
        inference_hash = make_param_hash(inference_config)
        inference_step_dir = os.path.join("artifacts", f"inference_{step}_{inference_hash}")
        
        # 3. Use training dir for loading artifacts
        step_dir = train_step_dir
        param_hash = train_hash

    manifest_file = os.path.join(step_dir, "manifest.json")

    if os.path.exists(manifest_file):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {step_dir}")
        self.paths[step] = step_dir
        self.hashes[step] = param_hash
        self.dataframes["test_num"] = pd.read_csv(os.path.join(step_dir, f"test_num_{param_hash}.csv"))
        if train_mode:
            self.dataframes["train_num"] = pd.read_csv(os.path.join(step_dir, f"train_num_{param_hash}.csv"))
            self.dataframes["val_num"] = pd.read_csv(os.path.join(step_dir, f"val_num_{param_hash}.csv"))
            #if excluded_df is not None:
            self.dataframes["excluded_num"] = pd.read_csv(os.path.join(step_dir, f"excluded_num_{param_hash}.csv")) if excluded_df is not None else None
        return

    os.makedirs(step_dir, exist_ok=True)

    if train_mode:
        train_numeric = train_df.copy()
        dropped = []
        encoded_mapping = {}
        inverse_mapping = {}
        grouping_map = {}
        id_like_columns = []

        # First, identify and drop constant columns (no variance or all null)
        constant_columns = []
        for col in train_numeric.columns:
            # Check for completely null columns
            if train_numeric[col].isna().all():
                constant_columns.append(col)
                grouping_map[col] = {
                    "strategy": "drop_constant",
                    "reason": "all_null"
                }
                continue
                
            # Check for numeric columns with zero variance
            if pd.api.types.is_numeric_dtype(train_numeric[col]):
                if train_numeric[col].nunique() <= 1:
                    constant_columns.append(col)
                    grouping_map[col] = {
                        "strategy": "drop_constant",
                        "reason": "zero_variance_numeric"
                    }
                    continue
            
            # Check for categorical columns with only one value
            else:
                if train_numeric[col].nunique() <= 1:
                    constant_columns.append(col)
                    grouping_map[col] = {"strategy": "drop_constant", "reason": "single_value_categorical"}
                    continue

        # Drop constant columns
        if constant_columns:
            train_numeric.drop(columns=constant_columns, inplace=True)
            dropped.extend(constant_columns)
            print(f"[{step.upper()}] Dropped {len(constant_columns)} constant columns: {constant_columns}")

        for col in train_df.columns:
            if col in constant_columns:
                continue
                
            if pd.api.types.is_numeric_dtype(train_df[col]):
                continue

            cardinality = train_df[col].nunique(dropna=False)
            col_fraction = cardinality / dataset_size

            if cardinality <= config["c1"]:
                # Low cardinality: keep and one-hot later
                continue

            elif col_fraction <= config["c2"]:
                # Mid cardinality: keep top C1 categories, rest → "Other"
                top_cats = train_df[col].value_counts().nlargest(config["c1"]).index
                train_numeric[col] = train_df[col].where(train_df[col].isin(top_cats), other="Other")
                grouping_map[col] = {
                    "strategy": "top_c1+other",
                    "top_categories": top_cats.tolist()
                }

            elif config["b1"]:
                # High cardinality but treating as mid
                top_cats = train_df[col].value_counts().nlargest(config["c1"]).index
                train_numeric[col] = train_df[col].where(train_df[col].isin(top_cats), other="Other")
                grouping_map[col] = {
                    "strategy": "high_as_mid",
                    "top_categories": top_cats.tolist()
                }

            elif config["id_like_exempt"]:
                log_ratio = np.log10(dataset_size) / np.log10(max(cardinality, 2))
                if 1 <= log_ratio <= config["c3"]:
                    id_like_columns.append(col)
                    dropped.append(col)
                    train_numeric.drop(columns=[col], inplace=True)
                    grouping_map[col] = {"strategy": "id_like_exempt"}
                    continue

                # otherwise drop
                dropped.append(col)
                grouping_map[col] = {"strategy": "drop"}

            else:
                dropped.append(col)
                train_numeric.drop(columns=[col], inplace=True)
                grouping_map[col] = {"strategy": "drop"}

        # Process val and test similar to train
        val_numeric = val_df.copy()
        excluded_numeric = excluded_df.copy() if excluded_df is not None else None
        # Apply same transformations to val and excluded
        

    test_numeric = test_df.copy()
    # Apply same transformations to val, test, and excluded
    for col in train_df.columns:
        if col in dropped:
            if col in val_numeric.columns:
                val_numeric.drop(columns=[col], inplace=True)
            if col in test_numeric.columns:
                test_numeric.drop(columns=[col], inplace=True)
            if excluded_numeric is not None and col in excluded_numeric.columns:
                excluded_numeric.drop(columns=[col], inplace=True)
        elif col in grouping_map:
            strategy = grouping_map[col]["strategy"]
            if strategy in ["top_c1+other", "high_as_mid"] and "top_categories" in grouping_map[col]:
                top_cats = grouping_map[col]["top_categories"]
                if col in val_numeric.columns:
                    val_numeric[col] = val_numeric[col].where(val_numeric[col].isin(top_cats), other="Other")
                if col in test_numeric.columns:
                    test_numeric[col] = test_numeric[col].where(test_numeric[col].isin(top_cats), other="Other")
                if excluded_numeric is not None and col in excluded_numeric.columns:
                    excluded_numeric[col] = excluded_numeric[col].where(excluded_numeric[col].isin(top_cats), other="Other")
    
    # Make sure val_numeric and test_numeric have the same columns as train_numeric
    # Keep only columns that are in train_numeric
    test_numeric = test_numeric[[col for col in test_numeric.columns if col in train_numeric.columns]]
    val_numeric = val_numeric[[col for col in val_numeric.columns if col in train_numeric.columns]]
    excluded_numeric = excluded_numeric[[col for col in excluded_numeric.columns if col in train_numeric.columns]] if excluded_numeric is not None else None

    # Handle missing values
    imputation_stats = {}
    
    # Get central tendency preference from config
    central_tendency = self.config.get("central_tendency", "median")  # Default to median if not specified
    
    # Process all datasets in parallel
    datasets = {
        "train": train_numeric,
        "val": val_numeric,
        "test": test_numeric,
        "excluded": excluded_numeric,
    }
    
    # Add excluded dataset if available
    if "excluded" in self.dataframes:
        excluded_df = self.dataframes["excluded"].copy()
        excluded_numeric = excluded_df.copy()
        
        # Apply same transformations to excluded dataset
        for col in train_df.columns:
            if col in dropped:
                if col in excluded_numeric.columns:
                    excluded_numeric.drop(columns=[col], inplace=True)
            elif col in grouping_map:
                strategy = grouping_map[col]["strategy"]
                if strategy in ["top_c1+other", "high_as_mid"] and "top_categories" in grouping_map[col]:
                    top_cats = grouping_map[col]["top_categories"]
                    if col in excluded_numeric.columns:
                        excluded_numeric[col] = excluded_numeric[col].where(excluded_numeric[col].isin(top_cats), other="Other")
        
        # Keep only columns that are in train_numeric
        excluded_numeric = excluded_numeric[[col for col in excluded_numeric.columns if col in train_numeric.columns]]
        datasets["excluded"] = excluded_numeric
    
    # For numeric columns, impute with central tendency
    numeric_cols = train_numeric.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Calculate central value based on config
        if central_tendency == "mean":
            central_value = train_numeric[col].mean()
        else:  # median
            central_value = train_numeric[col].median()
            
        imputation_stats[col] = {
            "strategy": central_tendency, 
            "value": central_value
        }
        
        # Impute missing values across all datasets
        for dataset_name, dataset in datasets.items():
            if col in dataset.columns:  # Replace these lines around line 263:
                dataset.loc[:, col] = dataset[col].fillna(central_value)  # dataset[col].fillna(central_value, inplace=True) 
    
    # For categorical columns, add indicator and impute with mode
    categorical_cols = train_numeric.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        # Create indicator columns for missing values within the column
        for dataset_name, dataset in datasets.items():
            if col in dataset.columns:
                # Add is_NA indicator for null values
                missing_mask = dataset[col].isna() | (dataset[col] == "")
                dataset[f"{col}_is_NA"] = missing_mask.astype(int)
        
        # Calculate mode on training data
        col_mode = train_numeric[col].mode()[0] if not train_numeric[col].mode().empty else "MISSING"
        imputation_stats[col] = {"strategy": "mode", "value": col_mode}
        
        # Impute missing values across all datasets
        for dataset_name, dataset in datasets.items():
            if col in dataset.columns:
                dataset.loc[:, col] = dataset[col].fillna(col_mode)
                dataset.loc[:, col] = dataset[col].replace("", col_mode)

    # One-hot encode categorical columns for all datasets
    categorical_cols = train_numeric.select_dtypes(include=["object", "category"]).columns.tolist()
    encoded_datasets = {}
    
    for dataset_name, dataset in datasets.items():
        encoded_datasets[dataset_name] = pd.get_dummies(dataset, columns=categorical_cols, drop_first=False)
    
    train_encoded = encoded_datasets["train"]
    val_encoded = encoded_datasets["val"]
    test_encoded = encoded_datasets["test"]
    if "excluded" in encoded_datasets:
        excluded_encoded = encoded_datasets["excluded"]
    
    # Ensure column consistency across all datasets
    for dataset_name, dataset in encoded_datasets.items():
        if dataset_name == "train":
            continue
            
        # Add missing columns with zeros
        for col in train_encoded.columns:
            if col not in dataset.columns:
                dataset[col] = 0
        
        # Remove extra columns
        extra_cols = [col for col in dataset.columns if col not in train_encoded.columns]
        if extra_cols:
            dataset.drop(columns=extra_cols, inplace=True)
        
        # Ensure same column order
        encoded_datasets[dataset_name] = dataset[train_encoded.columns]
    
    # Update datasets with encoded versions
    train_encoded = encoded_datasets["train"]
    val_encoded = encoded_datasets["val"]
    test_encoded = encoded_datasets["test"]
    if "excluded" in encoded_datasets:
        excluded_encoded = encoded_datasets["excluded"]

    numeric_train_csv = os.path.join(step_dir, f"train_num_{param_hash}.csv")
    numeric_val_csv = os.path.join(step_dir, f"val_num_{param_hash}.csv")
    numeric_test_csv = os.path.join(step_dir, f"test_num_{param_hash}.csv")

    if "excluded" in encoded_datasets:    
        numeric_excluded_csv = os.path.join(step_dir, f"excluded_num_{param_hash}.csv")
    

    train_encoded.to_csv(numeric_train_csv, index=False)
    val_encoded.to_csv(numeric_val_csv, index=False)
    test_encoded.to_csv(numeric_test_csv, index=False)
    if "excluded" in encoded_datasets:
        excluded_encoded.to_csv(numeric_excluded_csv, index=False)

    grouping_json = os.path.join(step_dir, f"grouping_map_{param_hash}.json")
    mapping_json = os.path.join(step_dir, f"encoded_mapping_{param_hash}.json")

    with open(grouping_json, "w") as f:
        grouping_map = convert_numpy_types(grouping_map)  # Convert NumPy types to native Python types
        json.dump(grouping_map, f, indent=2)
    with open(mapping_json, "w") as f:
        json.dump({
            "original_to_encoded": encoded_mapping,
            "encoded_to_original": inverse_mapping
        }, f, indent=2)

    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "output_dir": step_dir,
        "outputs": {
            "train_num_csv": numeric_train_csv,
            "val_num_csv": numeric_val_csv,
            "test_num_csv": numeric_test_csv,
            "grouping_map_json": grouping_json,
            "encoded_mapping_json": mapping_json
        }
    }
    
    # Update imputation stats to the manifest
    manifest["imputation_stats"] = imputation_stats
    imputation_json = os.path.join(step_dir, f"imputation_stats_{param_hash}.json")
    with open(imputation_json, "w") as f:
        json.dump(imputation_stats, f, indent=2)
    manifest["outputs"]["imputation_stats_json"] = imputation_json

    # Store detailed metadata about the transformations
    metadata = {
        "dropped_columns": dropped,
        "constant_columns": constant_columns if 'constant_columns' in locals() else [],
        "id_like_columns": id_like_columns,
        "grouping_map": grouping_map,
        "imputation_stats": imputation_stats,
        "original_columns": train_df.columns.tolist(),
        "retained_numeric_columns": train_numeric.select_dtypes(include=['number']).columns.tolist(),
        "retained_categorical_columns": train_numeric.select_dtypes(include=['object', 'category']).columns.tolist(),
        "created_is_NA_columns": [f"{col}_is_NA" for col in train_numeric.select_dtypes(include=['object', 'category']).columns],
        "encoded_columns": train_encoded.columns.tolist(),
        "encoding_mapping": encoded_mapping,
        "inverse_mapping": inverse_mapping,
        "datasets_processed": list(datasets.keys())
    }
    
    # Add metadata to manifest
    manifest["metadata"] = metadata
    metadata_json = os.path.join(step_dir, f"metadata_{param_hash}.json")
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)
    manifest["outputs"]["metadata_json"] = metadata_json
    
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    if self.config.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(step_dir, artifact_path=step)

    log_registry(step, param_hash, config, step_dir)

    # Store all artifacts and metadata in class state
    self.dataframes["test_num"] = test_encoded
    if train_mode:
        self.dataframes["train_num"] = train_encoded
        self.dataframes["val_num"] = val_encoded

        if "excluded" in encoded_datasets:
            self.dataframes["excluded_num"] = excluded_encoded
    
    # This section is redundant - the excluded dataset is saved again
    # if "excluded" in encoded_datasets:
    #     # Save excluded dataset
    #     excluded_numeric_csv = os.path.join(step_dir, f"excluded_num_{param_hash}.csv")
    #     excluded_encoded.to_csv(excluded_numeric_csv, index=False)  # This is redundant - already saved above
    #     # Add to manifest outputs
    #     manifest["outputs"]["excluded_num_csv"] = excluded_numeric_csv  # Already added earlier
        
    # if "excluded" in self.dataframes:
    #     # Ensure the excluded dataset has EXACTLY the same columns as train_encoded
    #     if "excluded" in encoded_datasets:
    #         # Make absolutely sure column order and presence matches exactly
    #         for col in train_encoded.columns:
    #             if col not in excluded_encoded.columns:
    #                 excluded_encoded[col] = 0  # Redundant - columns already aligned earlier
            
    #         # Remove any extra columns not in train_encoded
    #         extra_cols = [col for col in excluded_encoded.columns if col not in train_encoded.columns]
    #         if extra_cols:
    #             excluded_encoded.drop(columns=extra_cols, inplace=True)  # Redundant - already done in column consistency step
                
    #         # Ensure same column order as train_encoded
    #         excluded_encoded = excluded_encoded[train_encoded.columns]  # Redundant - already done in column consistency step
            
    #     self.dataframes["excluded_num"] = excluded_encoded  # Redundant - already stored earlier
    
    # Track all features and their transformations
    original_features = train_df.columns.tolist()
    current_features = train_encoded.columns.tolist()
    
    # Track dropped features
    all_dropped_features = dropped + constant_columns
    
    # Track transformed features mapping
    transformed_features = {}
    for col in train_df.select_dtypes(include=['object', 'category']).columns:
        if col not in all_dropped_features:
            # Find all one-hot encoded columns for this original feature
            one_hot_cols = [c for c in train_encoded.columns if c.startswith(f"{col}_")]
            transformed_features[col] = one_hot_cols
    
    # Register features for this stage
    # self.register_features("numeric_conversion", current_features)
    
    # Track changes
    # self.track_feature_changes( stage="numeric_conversion",dropped=all_dropped_features, transformed=transformed_features)
    
    # Store feature information explicitly
    feature_info = {
        "original_columns": original_features,
        "encoded_columns": current_features,
        "dropped_columns": all_dropped_features,
        "transformation_mapping": transformed_features
    }
    
    feature_info_file = os.path.join(step_dir, f"feature_info_{param_hash}.json")
    with open(feature_info_file, "w") as f:
        json.dump(feature_info, f, indent=2)
    manifest["outputs"]["feature_info_json"] = feature_info_file
        
    # Store all metadata in class state
    self.paths[step] = step_dir
    self.hashes[step] = param_hash
    self.artifacts[step] = manifest["outputs"]
    self.transformations[step] = {
        "grouping_map": grouping_map,
        "imputation_stats": imputation_stats,
        "feature_columns": train_encoded.columns.tolist()  # Store the complete column list
    }
    
    # Save the feature names explicitly for model training to ensure consistency
    feature_names_file = os.path.join(step_dir, f"feature_names_{param_hash}.json")
    with open(feature_names_file, "w") as f:
        json.dump({"feature_names": train_encoded.columns.tolist()}, f, indent=2)
    manifest["outputs"]["feature_names_json"] = feature_names_file


if __name__ == "__main__":
    # Example usage to test the numeric conversion functionality
    from ml_pipeline.base import MLPipeline
    import pandas as pd
    import numpy as np
    
    # Create mock datasets for testing
    np.random.seed(42)
    n_samples = 200
    
    # Create sample data with various column types
    mock_train = pd.DataFrame({
        "id": range(1, n_samples + 1),
        "numeric_col": np.random.normal(0, 1, n_samples),
        "integer_col": np.random.randint(0, 100, n_samples),
        "low_card_cat": np.random.choice(['A', 'B', 'C'], n_samples),
        "mid_card_cat": np.random.choice([f'Cat_{i}' for i in range(20)], n_samples),
        "high_card_cat": np.random.choice([f'ID_{i}' for i in range(150)], n_samples),
        "id_like_col": [f'U{i:05d}' for i in range(n_samples)],
        "missing_values_col": np.random.choice([np.nan, 1, 2, 3], n_samples),
        "target": np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    
    # Add some empty strings to test that case
    mock_train.loc[np.random.choice(n_samples, 10), "low_card_cat"] = ""
    
    # Create validation and test sets with similar structure but different values
    mock_val = mock_train.copy().sample(n=50, random_state=42)
    mock_val.reset_index(drop=True, inplace=True)
    mock_val["numeric_col"] = mock_val["numeric_col"] + np.random.normal(0, 0.1, len(mock_val))
    
    mock_test = mock_train.copy().sample(n=50, random_state=43)
    mock_test.reset_index(drop=True, inplace=True)
    mock_test["numeric_col"] = mock_test["numeric_col"] + np.random.normal(0, 0.1, len(mock_test))
    
    # Add some values in validation/test that weren't in training
    mock_val.loc[0, "mid_card_cat"] = "New_Category_Val"
    mock_test.loc[0, "mid_card_cat"] = "New_Category_Test"
    
    # Create a simple configuration for testing
    test_config = {
        "target_col": "target",
        "id_col": "id",
        "use_mlflow": False,
        "c1": 5,          # Cardinality threshold for one-hot encoding
        "c2": 0.2,        # Fraction threshold for rare category reduction
        "b1": True,       # Treat high-cardinality vars as mid if True
        "c3": 1.5,        # Log-scale threshold for ID-like columns
        "id_like_exempt": True,  # Whether to drop ID-like columns
        "central_tendency": "median"  # Use median for imputation
    }
    
    # Initialize the pipeline with test configuration
    pipeline = MLPipeline(config=test_config)
    
    # Add mock data to the pipeline state
    pipeline.dataframes = {
        "train": mock_train,
        "val": mock_val,
        "test": mock_test,
        "excluded": mock_train.sample(n=20, random_state=44)  # Create a small excluded set
    }
    
    # Run the numeric conversion step
    pipeline.numeric_conversion()
    
    # Display results summary
    print("\nNumeric Conversion Results Summary:")
    print("-" * 40)
    
    # Show shape changes
    print(f"Original train shape: {mock_train.shape}")
    print(f"Numeric train shape: {pipeline.dataframes['train_num'].shape}")
    
    # Show column type counts
    non_numeric_cols = mock_train.select_dtypes(exclude=['number']).columns
    print(f"\nOriginal non-numeric columns: {len(non_numeric_cols)}")
    print(f"Original non-numeric columns: {list(non_numeric_cols)}")
    
    # Show one-hot encoded columns
    numeric_cols = pipeline.dataframes['train_num'].columns
    print(f"\nFinal numeric columns: {len(numeric_cols)}")
    
    # Check for any missing values
    missing_counts = pipeline.dataframes['train_num'].isna().sum().sum()
    print(f"Missing values after imputation: {missing_counts}")
    
    # Show output directory and artifacts
    print(f"\nOutput directory: {pipeline.paths['numeric_conversion']}")
    print(f"Artifacts created: {list(pipeline.artifacts.get('numeric_conversion', {}).keys())}")
    
    # Verify consistency across datasets
    col_match = (set(pipeline.dataframes['train_num'].columns) == 
                 set(pipeline.dataframes['val_num'].columns) == 
                 set(pipeline.dataframes['test_num'].columns) == 
                 set(pipeline.dataframes['excluded_num'].columns))
    print(f"\nColumn consistency across all datasets: {col_match}")
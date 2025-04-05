from ml_pipeline.base import MLPipeline

if __name__ == "__main__":
    configs = [
        {
            "target_column": "is_fraud",  # Changed from target_col to match the expected key
            "id_column": "transaction_id", # Changed from id_col to match
            "random_state": 42,           # Changed from seed to match
            "train_size": 0.6,
            "val_size": 0.1,
            "test_size": 0.3,
            "stratify": True,             # Added to match expected parameter
            "stratify_cardinality_threshold": 10,
            "c1": 10,        # max categories to keep before 'Other'
            "c2": 0.01,      # fraction threshold for rare category reduction
            "b1": True,      # treat high as mid-cardinality
            "c3": 10,        # log-scale threshold for ID-like exemption
            "id_like_exempt": True,
            "use_mlflow": True
        }
    ]
    
    for config in configs:
        print(f"Running pipeline with configuration: {config}")
        pipeline = MLPipeline(config)
        
        # Debug the pipeline state
        print("[Debug] Initial pipeline state:", 
              f"Dataframes keys: {list(pipeline.dataframes.keys() if pipeline.dataframes else [])}")
        
        pipeline.run_all()
        
        print("[Debug] Final pipeline state:", 
              f"Dataframes keys: {list(pipeline.dataframes.keys() if pipeline.dataframes else [])}")
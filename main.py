from ml_pipeline import MLPipeline

if __name__ == "__main__":
    config = {
        "target_column": "is_fraud",
        "id_col": "transaction_id",
        "seed": 42,
        "train_size": 0.6,
        "val_size": 0.1,
        "test_size": 0.3,
        "stratify_cardinality_threshold": 10,
        "c1": 10,
        "c2": 0.01,
        "b1": True,
        "c3": 10,
        "id_like_exempt": True
    }

    pipeline = MLPipeline(config)
    pipeline.run_all()
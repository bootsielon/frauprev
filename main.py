from ml_pipeline.base import MLPipeline
from ml_pipeline.utils import make_param_hash


def main() -> None:
    configs = [
        {
            "seed": 42,
            "target_col": "is_fraud",
            "id_col": "transaction_id",
            "use_mlflow": True,
            
            "train_size": 0.6,
            "test_size": 0.3,
            "val_size": 0.1,
            "use_stratification": False,
            "use_downsampling": True,
            # "stratify_cols": ["client_id", "merchant_id"],
            "stratify_cardinality_threshold": 5,

            # Step 3
            "c1": 10,
            "c2": 0.01,
            "b1": True,
            "c3": 5,

            # Step 4
            "s1": False,
            "t1": False,

            # Step 8
            "shap_cutoff": 1 - 1e-5,  # 0.95,

            # Step 10
            "use_cluster_select": [True, False],
            "save_fs_mods": False,

            # Step 11
            "k_folds": 5,
            "opt_metric": "f1",
            "minimize_metric": False,

            # Step 12: Baseline hyperparameters
            "baseline_hyperparams": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "use_label_encoder": False,
                "eval_metric": "logloss"
            }
        },
        # Add more config dicts here
    ]

    for cfg in configs:
        param_hash = make_param_hash(cfg)
        print(f"\n=== Running pipeline for hash: {param_hash} ===")
        pipeline = MLPipeline(cfg)
        pipeline.run_all()


if __name__ == "__main__":
    main()
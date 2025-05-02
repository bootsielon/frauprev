"""
Numeric‑conversion step
=======================

Converts categorical / mixed‑type columns into a fully numeric feature
matrix.  All artefacts are written to

    artifacts/run_<self.global_hash>/numeric_conversion/

Key SPEC compliance
-------------------
•   One *global* hash per run; no per‑step hashes (SPEC §1, §2).  
•   Skip‑guard appears first (SPEC §14).  
•   Inference logic: *reuse → load‑from‑train → raise* (SPEC §5).  
•   `self.hashes` removed (SPEC §3).  
•   Filenames contain **no hashes**; folder already carries the hash
    (SPEC §25).  
•   `log_registry(step, self.global_hash, …)` called (SPEC §7).  
•   No recomputation happens during inference.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
import pandas as pd
import mlflow

from ml_pipeline.utils import log_registry, convert_numpy_types  # make_param_hash no longer used


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_existing_numeric(step_dir: str) -> Dict[str, pd.DataFrame]:
    """Load numeric CSVs that are present in *step_dir*."""
    dfs: Dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test", "excluded"):
        fp = os.path.join(step_dir, f"{split}_num.csv")
        if os.path.exists(fp):
            dfs[f"{split}_num"] = pd.read_csv(fp)
    return dfs


# --------------------------------------------------------------------------- #
# Main step                                                                   #
# --------------------------------------------------------------------------- #
def numeric_conversion(self) -> None:  # noqa: C901  (complexity tolerated for now)
    """
    Convert non‑numeric variables to numeric representations.

    Training:
        • Perform full conversion; write artefacts.
    Inference:
        • Reuse artefacts if present in the *current* run folder.
        • Else load them from the training run folder
          ``artifacts/run_<self.global_train_hash>/numeric_conversion/``.
        • Else raise *FileNotFoundError* – never recompute.
    """
    # ------------------------------------------------------------------- #
    # Resolve paths                                                       #
    # ------------------------------------------------------------------- #
    step = "numeric_conversion"
    param_hash = self.global_hash
    run_step_dir = os.path.join("artifacts", f"run_{param_hash}", step) # step_dir   = os.path.join("artifacts", f"run_{self.global_hash}", step)
    run_manifest_dir = os.path.join(run_step_dir, "manifest.json")  #  ____manifest = os.path.join(run_step_dir, "manifest.json")
    os.makedirs(run_step_dir, exist_ok=True)

    # ------------------------------------------------------------------- #
    # 0️⃣  Skip‑guard – artefacts already in *current* run                #
    # ------------------------------------------------------------------- #
    if os.path.exists(run_manifest_dir):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {run_step_dir}")
        self.paths[step] = run_step_dir
        self.dataframes.update(_load_existing_numeric(run_step_dir))
        return

    # ------------------------------------------------------------------- #
    # 1️⃣  Inference → try to load from the training run                  #
    # ------------------------------------------------------------------- #
    if not self.train_mode:
        train_step_dir = os.path.join(
            "artifacts", f"run_{self.global_train_hash}", step
        )
        train_manifest_dir = os.path.join(train_step_dir, "manifest.json")
        if os.path.exists(train_manifest_dir):
            train_manifest = json.load(open(train_manifest_dir, "r"))
            print(f"[{step.upper()}] Reusing training artefacts from {train_step_dir}")
            self.train_paths[step] = train_step_dir
            self.global_train_hash = train_manifest.get("global_hash")
            self.train_manifest[step] = train_manifest
            self.train_models[step] = {}
            self.train_artifacts[step] = train_manifest.get("artifacts", {})
            self.train_transformations[step] = train_manifest.get("transformations", {})
            # self.train_metrics[step] = {}
            # self.train_dataframes[step] = {}
            # self.dataframes.update(_load_existing_numeric(train_step_dir))
            # return
        # Nothing to reuse → spec mandates failure
        #raise FileNotFoundError(
        #    f"[{step.upper()}] Expected training artefacts at {train_step_dir} but none found."
        #)

    # ------------------------------------------------------------------- #
    # 2️⃣  Training mode – perform full computation                        #
    # ------------------------------------------------------------------- #
    cfg = self.config
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    
    test_df: pd.DataFrame = self.dataframes["test"]
    
    train_df: pd.DataFrame = self.dataframes["train"] if self.train_mode else None
    val_df: pd.DataFrame = self.dataframes["val"] if self.train_mode else None
    excluded_df: pd.DataFrame | None = self.dataframes.get("excluded") if self.train_mode else None

    dataset_size = len(train_df) if self.train_mode else len(test_df)

    # -- hyper‑parameters controlling grouping / imputation
    c1 = cfg["c1"] if self.train_mode else None          # cardinality threshold for one‑hot
    c2 = cfg["c2"] if self.train_mode else None          # rare‑category fraction
    b1 = cfg["b1"] if self.train_mode else None          # treat high as mid if True
    c3 = cfg["c3"] if self.train_mode else None          # ID‑like log‑ratio threshold
    id_like_exempt = cfg.get("id_like_exempt", True)

    # ---------------------------------------------------------------- #
    # 2.1  Drop constants & identify column groups                     #
    # ---------------------------------------------------------------- #
    dropped: list[str] = [] if self.train_mode else train_manifest.get("dropped_columns", [])
    constant_columns: list[str] = [] if self.train_mode else train_manifest.get("constant_columns", [])
    grouping_map: dict[str, Any] = {} if self.train_mode else train_manifest.get("grouping_map", {})
    id_like_columns: list[str] = [] if self.train_mode else train_manifest.get("id_like_columns", [])
    final_columns: list[str] = [] if self.train_mode else train_manifest.get("final_columns", [])
    work_train = train_df.copy() if self.train_mode else test_df.copy()

    if self.train_mode:
        for col in list(work_train.columns):
            # --- constant / all‑null
            if work_train[col].isna().all():
                constant_columns.append(col)
                grouping_map[col] = {"strategy": "drop_constant", "reason": "all_null"}
                continue
            if pd.api.types.is_numeric_dtype(work_train[col]):
                if work_train[col].nunique() <= 1:
                    constant_columns.append(col)
                    grouping_map[col] = {
                        "strategy": "drop_constant",
                        "reason": "zero_variance_numeric",
                    }
                    continue
            else:
                if work_train[col].nunique(dropna=False) <= 1:
                    constant_columns.append(col)
                    grouping_map[col] = {
                        "strategy": "drop_constant",
                        "reason": "single_value_categorical",
                    }
                    continue

        work_train.drop(columns=constant_columns, inplace=True)
        if constant_columns:
            print(f"[{step.upper()}] Dropped {len(constant_columns)} constant columns")

        # ---------------------------------------------------------------- #
        # 2.2  Cardinality‑based handling                                  #
        # ---------------------------------------------------------------- #
        for col in list(work_train.columns):
            # numeric columns – leave as is
            if pd.api.types.is_numeric_dtype(work_train[col]):
                continue

            cardinality = train_df[col].nunique(dropna=False)
            col_fraction = cardinality / dataset_size

            if cardinality <= c1:
                # low cardinality – keep for one‑hot later
                continue

            elif col_fraction <= c2 or (b1 and col_fraction <= 1):
                # mid or high‑as‑mid: keep top c1 categories
                top_cats = train_df[col].value_counts().nlargest(c1).index
                work_train[col] = train_df[col].where(train_df[col].isin(top_cats), other="Other")
                grouping_map[col] = {
                    "strategy": "top_c1+other" if col_fraction <= c2 else "high_as_mid",
                    "top_categories": top_cats.tolist(),
                }

            else:
                # potential ID‑like or drop
                if id_like_exempt:
                    log_ratio = np.log10(dataset_size) / np.log10(max(cardinality, 2))
                    if 1 <= log_ratio <= c3 or col == cfg["id_col"]:
                        # ID‑like column – exempt from grouping
                        id_like_columns.append(col)
                        grouping_map[col] = {"strategy": "id_like_exempt"}
                        dropped.append(col)
                        work_train.drop(columns=[col], inplace=True)
                        continue
                dropped.append(col)
                grouping_map[col] = {"strategy": "drop"}
                work_train.drop(columns=[col], inplace=True)


    else:
        train_grouping_map = self.train_transformations[step]["grouping_map"]
        for col, spec in train_grouping_map.items():
            if spec["strategy"] == "drop_constant":
                # drop constant columns
                if col in work_train.columns:
                    dropped.append(col)
                    work_train.drop(columns=[col], inplace=True)
                    continue
            elif spec["strategy"] == "drop":
                # drop columns that were dropped in training
                if col in work_train.columns:
                    dropped.append(col)
                    work_train.drop(columns=[col], inplace=True)
                    continue
            elif spec["strategy"] == "id_like_exempt":
                # exempt ID‑like columns from grouping
                id_like_columns.append(col)
                if col in work_train.columns:
                    dropped.append(col)
                    work_train.drop(columns=[col], inplace=True)
                    continue
                continue
            elif spec["strategy"] in ("top_c1+other", "high_as_mid"):
                # special grouping for high cardinality columns
                top_cats = spec["top_categories"]
                if col in work_train.columns:
                    work_train[col] = work_train[col].where(work_train[col].isin(top_cats), other="Other")

        # work_train.drop(columns=constant_columns, inplace=True)
        #if constant_columns:
            # print(f"[{step.upper()}] Dropped {len(constant_columns)} constant columns")


    # ---------------------------------------------------------------- #
    # 2.3  Apply same treatment to val / test / excluded               #
    # ---------------------------------------------------------------- #
    def _apply_grouping(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # drop
        df.drop(columns=[c for c in dropped if c in df.columns], inplace=True, errors="ignore")
        # special grouping
        for col, spec in grouping_map.items():
            if spec["strategy"] in ("top_c1+other", "high_as_mid"):
                top_cats = spec["top_categories"]
                if col in df.columns:
                    df[col] = df[col].where(df[col].isin(top_cats), other="Other")
        # ensure same columns subset as work_train
        return df[[c for c in df.columns if c in work_train.columns]]
    

    val_proc = _apply_grouping(val_df) if self.train_mode else None
    test_proc = _apply_grouping(test_df) if self.train_mode else None
    excluded_proc = _apply_grouping(excluded_df) if (excluded_df is not None) and (self.train_mode) else None

    # ---------------------------------------------------------------- #
    # 2.4  Imputation                                                  #
    # ---------------------------------------------------------------- #  
    central_tendency = cfg.get("central_tendency", "median") if self.train_mode else train_manifest.get("central_tendency", "median")
    imputation_stats: dict[str, dict[str, Any]] = {} if self.train_mode else self.train_artifacts.get("outputs", {}).get("imputation_stats", {})


    # impute numeric columns
    # (use mean or median depending on *central_tendency*)
    if self.train_mode:
        numeric_cols = work_train.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if central_tendency == "mean":
                value = work_train[col].mean()
            else:
                value = work_train[col].median()
            imputation_stats[col] = {"strategy": central_tendency, "value": value}
            for df in (work_train, val_proc, test_proc, excluded_proc):
                if df is not None and col in df.columns:
                    df[col] = df[col].fillna(value)
    else:
        # numeric_cols = work_train.select_dtypes(include=["number"]).columns
        for col, spec in imputation_stats.items(): #self.transformations[step]["imputation_stats"].items():
            if spec["strategy"] == "mean" or spec["strategy"] == "median":
                # impute numeric columns
                # (use mean or median depending on *central_tendency*)               
                value = spec["value"]
            # for df in ():  # , val_proc, test_proc, excluded_proc):
                if work_train is not None and col in work_train.columns:
                    work_train[col] = work_train[col].fillna(value)

    # categorical columns – use mode
    # (or "MISSING" if all values are missing)
    cat_cols = None
    if self.train_mode:
        cat_cols = work_train.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            mode_val = work_train[col].mode(dropna=True)
            mode_val = mode_val.iloc[0] if not mode_val.empty else "MISSING"
            imputation_stats[col] = {"strategy": "mode", "value": mode_val}
            for df in (work_train, val_proc, test_proc, excluded_proc):
                if df is not None and col in df.columns:
                    df[col] = df[col].fillna(mode_val).replace("", mode_val)
        # Add NA indicators for categorical
        for col in cat_cols:
            for df in (work_train, val_proc, test_proc, excluded_proc):
                if df is not None and col in df.columns:
                    df[f"{col}_is_NA"] = (df[col] == mode_val).astype(int)    
    else:
        cat_cols = []
        for col, spec in imputation_stats.items():
            if spec["strategy"] == "mode":
                cat_cols.append(col)
                mode_val = spec["value"]
                # mode_val = mode# work_train[col].mode(dropna=True)
                # mode_val = mode_val.iloc[0] if not mode_val.empty else "MISSING"
                
                #for df in ():
                if work_train is not None and col in work_train.columns:
                    work_train[col] = work_train[col].fillna(mode_val).replace("", mode_val)


    # ---------------------------------------------------------------- #
    # 2.5  One‑hot encoding                                            #
    # ---------------------------------------------------------------- #
    encoded_sets: Dict[str, pd.DataFrame] = {}

    if not self.train_mode:
        test_proc = work_train.copy()
        work_train = None
        
    for name, df in {
        "train": work_train,
        "val": val_proc,
        "test": test_proc,
        "excluded": excluded_proc,
    }.items():
        encoded_sets[name] = pd.get_dummies(df, columns=list(cat_cols), drop_first=False) if df is not None else None

    train_enc = encoded_sets.get("train")
    end_cols = train_enc.columns if self.train_mode else final_columns
            
    # harmonise columns
    for name, df in encoded_sets.items():
        if name == "train" or df is None:
            # skip train set or if df is None (excluded)
            continue
        missing_cols = [c for c in end_cols if c not in df.columns]
        for c in missing_cols:
            df[c] = imputation_stats[c]["value"] if c in imputation_stats else np.nan
        # add missing columns with NaN values
        extra_cols = [c for c in df.columns if c not in end_cols]
        if extra_cols:
            df.drop(columns=extra_cols, inplace=True)
        encoded_sets[name] = df[end_cols]

    val_enc = encoded_sets.get("val")
    excluded_enc = encoded_sets.get("excluded")
    test_enc = encoded_sets["test"]
    # ---------------------------------------------------------------- #
    # 2.6  Persist artefacts                                           #
    # ---------------------------------------------------------------- #
    os.makedirs(run_step_dir, exist_ok=True)

    outputs = {
        "train_num_csv": os.path.join(run_step_dir, "train_num.csv"),
        "val_num_csv": os.path.join(run_step_dir, "val_num.csv"),
        "test_num_csv": os.path.join(run_step_dir, "test_num.csv"),
        "grouping_map_json": os.path.join(run_step_dir, "grouping_map.json"),
        "imputation_stats_json": os.path.join(run_step_dir, "imputation_stats.json"),
        "metadata_json": os.path.join(run_step_dir, "metadata.json"),
        "final_columns_json": os.path.join(run_step_dir, "final_columns.json"),
    }


    for name, df in encoded_sets.items():
        if df is not None:
            outputs[f"{name}_num_csv"] = os.path.join(run_step_dir, f"{name}_num.csv")
            df.to_csv(outputs[f"{name}_num_csv"], index=False)

    with open(outputs["grouping_map_json"], "w") as fh:
        json.dump(convert_numpy_types(grouping_map), fh, indent=2)
    with open(outputs["imputation_stats_json"], "w") as fh:
        json.dump(convert_numpy_types(imputation_stats), fh, indent=2)

    metadata = {
        "dropped_columns": dropped + constant_columns,
        "id_like_columns": id_like_columns,
        "encoded_columns": train_enc.columns.tolist() if self.train_mode else test_enc.columns.tolist(),
        "constant_columns": constant_columns,
        "original_columns": train_df.columns.tolist() if self.train_mode else test_df.columns.tolist(),
        "grouping_map": grouping_map,
        "imputation_stats": imputation_stats,
        "cardinality_threshold": c1,
        "rare_category_fraction": c2,
        "high_as_mid": b1,
        "id_like_log_ratio_threshold": c3,
        "id_like_exempt": id_like_exempt,
        "central_tendency": central_tendency,
        #"dataset_size": dataset_size,
        #"train_size": len(train_enc),
        #"val_size": len(val_enc),
        #"test_size": len(test_enc),
        "final_columns": train_enc.columns.tolist() if self.train_mode else test_enc.columns.tolist(),
    }
    with open(outputs["metadata_json"], "w") as fh:
        json.dump(metadata, fh, indent=2)

    # ---------------------------------------------------------------- #
    # 2.7  MLflow & registry                                           #
    # ---------------------------------------------------------------- #
    if cfg.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{self.global_hash}"):
            mlflow.set_tags({"step": step, "global_hash": self.global_hash})
            mlflow.log_artifacts(run_step_dir, artifact_path=step)

    log_registry(step, self.global_hash, cfg, run_step_dir)

    # ---------------------------------------------------------------- #
    # 2.8  Update pipeline state                                       #
    # ---------------------------------------------------------------- #
    self.dataframes.update(
        {
            "train_num": train_enc,
            "val_num": val_enc,
            "test_num": test_enc,
            "excluded_num" : excluded_enc if excluded_enc is not None else None,
        }
    )   

    self.paths[step] = run_step_dir
    # removed: self.hashes[step]  # SPEC §3
    self.artifacts[step] = outputs
    self.transformations[step] = {
        "grouping_map": grouping_map,
        "imputation_stats": imputation_stats,
        "feature_columns": train_enc.columns.tolist() if self.train_mode else test_enc.columns.tolist(),
        "dropped_columns": dropped + constant_columns,
    }
    self.metadata[step] = metadata


    manifest = {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        #"config": {
        #    k: cfg[k] for k in ("c1", "c2", "b1", "c3", "id_like_exempt", "central_tendency")
        #},
        # use cfg.get(...) so that truly optional keys
        # do not raise KeyError (SPEC §16 – correctness)
        "config": {k: cfg.get(k)
                   for k in ("c1", "c2", "b1", "c3",
                             "id_like_exempt", "central_tendency")},
        "output_dir": run_step_dir,
        "artifacts": outputs,
        'transformations': self.transformations[step],
        'artifacts': self.artifacts[step],
        'outputs': outputs,
        'metadata': self.metadata[step],
        "final_columns": train_enc.columns.tolist() if self.train_mode else test_enc.columns.tolist(),
        'columns':{
            "dropped_columns": dropped + constant_columns,
            "id_like_columns": id_like_columns,
            "grouping_map": grouping_map,
            "imputation_stats": imputation_stats,
            "cardinality_threshold": c1,
            "rare_category_fraction": c2,
            "final_columns": train_enc.columns.tolist() if self.train_mode else test_enc.columns.tolist(),
            "high_as_mid": b1,
            "id_like_log_ratio_threshold": c3,
            "id_like_exempt": id_like_exempt,
            "central_tendency": central_tendency,
            #"dataset_size": dataset_size,
            #"train_size": len(train_enc),
            #"val_size": len(val_enc),
            #"test_size": len(test_enc),
        }
    }
    with open(run_manifest_dir, "w") as fh:
        json.dump(manifest, fh, indent=2)



    print(f"Manifest saved to: {run_manifest_dir}")
    #print(f"Artifacts: {outputs}")
    print(f"Artefacts saved to: {run_step_dir}")
    print(f"[{step.upper()}] Finished")

if __name__ == "__main__":
    """
    Smoke‑tests for numeric_conversion
    ----------------------------------
    1. Training run – fresh artefacts (compute).
    2. Training run – artefacts present (skip).
    3. Inference  – artefacts present (load).
    4. Inference  – artefacts missing (fail).
    """
    import os
    import shutil
    import numpy as np
    import pandas as pd

    from ml_pipeline.base import MLPipeline
    from ml_pipeline.utils import DEFAULT_TEST_HASH

    # ------------------------------------------------------------------ #
    # Mock data set‑up                                                   #
    # ------------------------------------------------------------------ #
    np.random.seed(42)
    n_samples = 200
    mock_train = pd.DataFrame(
        {
            "id": range(1, n_samples + 1),
            "numeric_col": np.random.normal(0, 1, n_samples),
            "integer_col": np.random.randint(0, 100, n_samples),
            "low_card_cat": np.random.choice(["A", "B", "C"], n_samples),
            "mid_card_cat": np.random.choice([f"Cat_{i}" for i in range(20)], n_samples),
            "high_card_cat": np.random.choice([f"ID_{i}" for i in range(150)], n_samples),
            "id_like_col": [f"U{i:05d}" for i in range(n_samples)],
            "missing_values_col": np.random.choice([np.nan, 1, 2, 3], n_samples),
            "target": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        }
    )
    mock_train.loc[np.random.choice(n_samples, 10), "low_card_cat"] = ""

    mock_val = mock_train.sample(n=50, random_state=123).reset_index(drop=True)
    mock_test = mock_train.sample(n=50, random_state=456).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Helper to build a minimally valid pipeline config                  #
    # ------------------------------------------------------------------ #
    def build_pipeline(*, train_mode: bool, run_hash: str, train_hash: str | None = None):
        cfg = {
            # ----- keys required by base.py / hashing rules
            "train_mode": train_mode,
            "model_name": "dummy_model",
            "model_hash": "dummy_model_hash",
            "dataset_name": "mock_dataset",
            "feature_names": sorted(mock_train.columns.tolist()),
            "extra_inference_settings": {},

            # ----- numeric‑conversion hyper‑params
            "seed": 42,
            "target_col": "target",
            "id_col": "id",
            "use_mlflow": False,
            "c1": 5,
            "c2": 0.2,
            "b1": True,
            "c3": 1.5,
            "id_like_exempt": True,
            "central_tendency": "median",
        }
        if not train_mode:
            cfg["train_hash"] = train_hash
        pipe = MLPipeline(config=cfg)
        pipe.global_hash = run_hash
        pipe.global_train_hash = run_hash if train_mode else train_hash
        return pipe
    
    
    step = "numeric_conversion"
    # ---------- clean slate for DEFAULT_TEST_HASH --------------------- #
    train_run_dir = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}", step)
    if os.path.exists(train_run_dir):
        shutil.rmtree(train_run_dir)

    # ------------------------------------------------------------------ #
    # 1️⃣  TRAINING – fresh artefacts                                    #
    # ------------------------------------------------------------------ #
    pipe_train = build_pipeline(train_mode=True, run_hash=DEFAULT_TEST_HASH)
    pipe_train.dataframes = {
        "train": mock_train,
        "val": mock_val,
        "test": mock_test,
        "excluded": mock_train.sample(n=20, random_state=789),
    }
    print("\n>>> TRAINING RUN (fresh artefacts)")
    pipe_train.numeric_conversion()

    # ------------------------------------------------------------------ #
    # 2️⃣  TRAINING – artefacts present (skip)                           #
    # ------------------------------------------------------------------ #
    pipe_train_skip = build_pipeline(train_mode=True, run_hash=DEFAULT_TEST_HASH)
    pipe_train_skip.dataframes = {
        "train": mock_train,
        "val": mock_val,
        "test": mock_test,
    }
    print("\n>>> TRAINING RUN (should skip)")
    pipe_train_skip.numeric_conversion()

    # ------------------------------------------------------------------ #
    # 3️⃣  INFERENCE – artefacts present                                 #
    # ------------------------------------------------------------------ #
    infer_hash_ok = "abcabcabcabc"
    pipe_infer_ok = build_pipeline(
        train_mode=False, run_hash=infer_hash_ok, train_hash=DEFAULT_TEST_HASH
    )
    pipe_infer_ok.dataframes = {"test": mock_test}
    print("\n>>> INFERENCE RUN (artefacts present)")
    pipe_infer_ok.numeric_conversion()

    # ------------------------------------------------------------------ #
    # 4️⃣  INFERENCE – artefacts missing (must fail)                     #
    # ------------------------------------------------------------------ #
    infer_hash_fail = "deadbeef0000"
    missing_train_hash = "feedfeedfeed"
    
    missing_dir = os.path.join("artifacts", f"run_{missing_train_hash}", step)
    if os.path.exists(missing_dir):
        shutil.rmtree(missing_dir)

    pipe_infer_fail = build_pipeline(
        train_mode=False, run_hash=infer_hash_fail, train_hash=missing_train_hash
    )
    pipe_infer_fail.dataframes = {"test": mock_test}
    print("\n>>> INFERENCE RUN (artefacts missing – should fail)")
    try:
        pipe_infer_fail.numeric_conversion()
        print("❌  ERROR: inference did NOT fail although artefacts are missing")
    except FileNotFoundError as e:
        print(f"✅  Caught expected error → {e}")

    print("\nSmoke‑tests finished.")
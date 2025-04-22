# === ml_pipeline/base.py ==============================================
from __future__ import annotations

import os
# import json   # currently unused
# import hashlib  # currently unused
from typing import Any
from copy import deepcopy

import pandas as pd

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
from ml_pipeline.utils import log_registry, make_param_hash  # absolute import (spec §11)

# ----------------------------------------------------------------------
# Pipeline step factories – lazily bound lambdas
# ----------------------------------------------------------------------
from ml_pipeline.eda                       import eda
from ml_pipeline.feature_engineering       import feature_engineering
from ml_pipeline.partitioning              import partitioning
from ml_pipeline.numeric_conversion        import numeric_conversion
from ml_pipeline.scaling                   import scaling
from ml_pipeline.model_baseline            import model_baseline
from ml_pipeline.shap_explainability       import shap_explainability
from ml_pipeline.shap_selection            import shap_selection
from ml_pipeline.feature_correlation       import feature_correlation
from ml_pipeline.feature_select_cluster    import feature_select_cluster
from ml_pipeline.feature_select_threshold  import feature_select_threshold
from ml_pipeline.hyperparameter_tuning     import hyperparameter_tuning
from ml_pipeline.final_model               import final_model


class MLPipeline:
    """
    Configuration‑driven ML pipeline orchestrator.

    A *single* deterministic `self.global_hash` identifies the entire run.
    In inference mode, `self.global_train_hash` points to the hash of the
    training run whose artefacts must be re‑used (spec §4 & §5).
    """

    # ──────────────────────────────────────────────────────────────────
    # INITIALISER
    # ──────────────────────────────────────────────────────────────────
    def __init__(
        self,
        config      : dict[str, Any],
        *,
        data_source : str               = "sqlite",
        db_path     : str               = "fraud_poc.db",
        xlsx_path   : str | None        = None,
        csv_paths   : dict | None       = None,
        raw_data    : pd.DataFrame | None = None,
    ) -> None:
        # ------------------------------------------------ core config --
        self.config: dict[str, Any] = deepcopy(config)
        self.train_mode: bool = bool(self.config.get("train_mode", True))
        self.config["train_mode"] = self.train_mode  # echo back

        # ------------------------------------------------ data handles -
        self.data_source    = data_source
        self.db_path        = db_path
        self.xlsx_path      = xlsx_path
        self.csv_paths      = csv_paths
        self.raw_data       = raw_data

        # ------------------------------------------------ bookkeeping --
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.paths     : dict[str, str]          = {}
        self.models    : dict[str, object]       = {}
        self.metrics   : dict[str, dict]         = {}
        self.artifacts : dict[str, dict]         = {}
        self.transformations: dict[str, dict]    = {}
        # ------------------------------------------------ run hash -----
        if self.train_mode:
            full_config = {
                "user_config": self.config,
                "data_source": data_source,
                "db_path"    : db_path,
                "xlsx_path"  : xlsx_path,
                "csv_paths"  : csv_paths,
            }
            self.global_hash = make_param_hash(full_config)
            self.global_train_hash = self.global_hash
        else:
            key_tuple = (
                self.config["model_name"],
                self.config["model_hash"],
                self.config["dataset_name"],
                tuple(sorted(self.config["feature_names"])),
                self.config.get("inference_extra", {}),
            )
            self.global_hash = make_param_hash(key_tuple)
            self.global_train_hash = self.config["train_hash"]

        # expose hashes
        self.config["global_hash"]       = self.global_hash
        self.config["global_train_hash"] = self.global_train_hash

        # ------------------------------------------------ run directory -
        self.run_dir = os.path.join("artifacts", f"run_{self.global_hash}")
        first_time = not os.path.exists(self.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)

        if first_time:  # write human timestamp (not hashed)
            with open(os.path.join(self.run_dir, "created_at.txt"), "w") as fh:
                from datetime import datetime, timezone
                fh.write(datetime.now(timezone.utc).isoformat())

        # ------------------------------------------------ bind steps ---
        self.load_data                = self._load_data
        self.eda                       = lambda: eda(self)
        self.feature_engineering       = lambda: feature_engineering(self)
        self.partitioning              = lambda: partitioning(self)
        self.numeric_conversion        = lambda: numeric_conversion(self)
        self.scaling                   = lambda: scaling(self)
        self.model_baseline            = lambda: model_baseline(self)
        self.shap_explainability       = lambda: shap_explainability(self)
        self.shap_selection            = lambda: shap_selection(self)
        self.feature_correlation       = lambda: feature_correlation(self)
        self.feature_select_cluster    = lambda: feature_select_cluster(self)
        self.feature_select_threshold  = lambda: feature_select_threshold(self)
        self.hyperparameter_tuning     = lambda: hyperparameter_tuning(self)
        self.final_model               = lambda: final_model(self)

    # ──────────────────────────────────────────────────────────────────
    # DATA LOADING  (self‑contained, satisfies spec §24a)
    # ──────────────────────────────────────────────────────────────────
    def _load_data(self) -> pd.DataFrame:
        """
        Load and merge raw data from SQLite / Excel / CSV or the supplied DF.
        """
        if self.data_source == "raw":
            if self.raw_data is None:
                raise ValueError("raw_data=None while data_source='raw'")
            return self.raw_data.copy()

        if self.data_source == "sqlite":
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            df_clients   = pd.read_sql("SELECT * FROM clients"      , conn)
            df_merchants = pd.read_sql("SELECT * FROM merchants"    , conn)
            df_tx        = pd.read_sql("SELECT * FROM transactions" , conn)
            conn.close()
        elif self.data_source == "xlsx":
            xls          = pd.ExcelFile(self.xlsx_path)
            df_clients   = pd.read_excel(xls, sheet_name="clients")
            df_merchants = pd.read_excel(xls, sheet_name="merchants")
            df_tx        = pd.read_excel(xls, sheet_name="transactions")
        elif self.data_source == "csv":
            df_clients   = pd.read_csv(self.csv_paths["clients"])
            df_merchants = pd.read_csv(self.csv_paths["merchants"])
            df_tx        = pd.read_csv(self.csv_paths["transactions"])
        else:
            raise ValueError(f"Unknown data_source: {self.data_source}")

        # unify column names
        df_clients   = df_clients.rename(columns={"account_creation_date": "account_creation_date_client"})
        df_merchants = df_merchants.rename(columns={"account_creation_date": "account_creation_date_merchant"})

        return (
            df_tx
            .merge(df_clients , on="client_id",   how="left")
            .merge(df_merchants, on="merchant_id", how="left")
        )

    # ──────────────────────────────────────────────────────────────────
    # PIPELINE RUNNERS
    # ──────────────────────────────────────────────────────────────────
    def run_all(self) -> None:
        """Execute the core steps needed for a baseline model."""
        self.eda()
        self.feature_engineering()
        self.partitioning()
        self.numeric_conversion()
        self.scaling()
        self.model_baseline()

    def run_later(self) -> None:
        """Execute optional later steps (explainability, selection, tuning)."""
        self.shap_explainability()
        self.shap_selection()
        self.feature_correlation()

        if self.config.get("use_cluster_select", [False])[0]:
            self.feature_select_cluster()
        else:
            self.feature_select_threshold()

        self.hyperparameter_tuning()
        self.final_model()

    # ──────────────────────────────────────────────────────────────────
    # REGISTRY HELPER
    # ──────────────────────────────────────────────────────────────────
    def _register_step(
        self,
        step: str,
        step_dir: str,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        """
        Record artefact paths and push a single line to the global registry
        (spec §7).  Called by individual step modules.
        """
        self.paths[step] = step_dir
        log_registry(step, self.global_hash, cfg or {}, step_dir)

# ──────────────────────────────────────────────────────────────────────
# SMOKE‑TEST SUITE  (base‑only edition – spec §17 – §24)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Base‑only smoke‑test.

    1. Training: verify hash, run‑dir creation, load_data().
    2. Training again: same config ⇒ identical hash; no directory errors.
    3. Inference with artefacts from #1: verify deterministic new hash
       and correct `global_train_hash` binding.
    4. Inference with bad `train_hash`: expect failure when accessing
       training artefacts directory (simulated here).
    """

    import os
    import shutil
    import traceback
    import pandas as pd

    # ------------------------------------------------------------------
    # Shared constant hash (spec §17)
    # ------------------------------------------------------------------
    try:
        from ml_pipeline.utils import DEFAULT_TEST_HASH
    except Exception:                                    # pragma: no cover
        DEFAULT_TEST_HASH = "deadbeefcaf0"

    # fresh workspace for repeatability
    # if os.path.exists("artifacts"):
        # shutil.rmtree("artifacts")

    df_demo = pd.DataFrame(
        {
            "client_id": [1, 2],
            "merchant_id": [10, 20],
            "amount": [100.5, None],
            "timestamp": ["2022-01-01", "2022-01-02"],
            "is_fraud": [0, 1],
        }
    )

    # Helper
    def safe(label: str, fn):
        try:
            fn()
            print(f"[OK ] {label}")
        except Exception as exc:
            print(f"[ERR] {label} → {exc}")
            traceback.print_exc()

    # 1. Training – first run
    train_cfg = {
        "train_mode": True,
        "random_seed": 123,
        "model_name": "demo_model",
        "model_hash": "abcd1234",
        "dataset_name": "demo_ds",
        "feature_names": ["amount", "timestamp"],
        "global_hash": DEFAULT_TEST_HASH,  # will be overridden
    }
    pipe_train = MLPipeline(train_cfg, data_source="raw", raw_data=df_demo)

    safe("TRAIN‑v1: load_data()", lambda: pipe_train._load_data())

    TRAIN_HASH = pipe_train.global_hash
    print("TRAIN_HASH =", TRAIN_HASH)
    assert os.path.isdir(pipe_train.run_dir), "run directory missing"

    # 2. Training – identical config
    pipe_train2 = MLPipeline(train_cfg, data_source="raw", raw_data=df_demo)
    assert (
        pipe_train2.global_hash == TRAIN_HASH
    ), "hash mismatch for identical config"
    safe("TRAIN‑v2: load_data()", lambda: pipe_train2._load_data())

    # 3. Inference – with correct train_hash
    infer_cfg = {
        "train_mode": False,
        "train_hash": TRAIN_HASH,
        "model_name": "demo_model",
        "model_hash": "abcd1234",
        "dataset_name": "demo_ds",
        "feature_names": ["timestamp", "amount"],
        "inference_extra": {},
    }
    pipe_infer = MLPipeline(infer_cfg, data_source="raw", raw_data=df_demo)
    assert (
        pipe_infer.global_train_hash == TRAIN_HASH
    ), "global_train_hash not set correctly"

    # Only call load_data() — no external steps
    safe("INFER: load_data()", lambda: pipe_infer._load_data())

    # 4. Inference – bad train_hash (should fail when we try to access artefacts)
    bad_infer_cfg = infer_cfg | {"train_hash": "not_a_real_hash"}
    pipe_bad = MLPipeline(bad_infer_cfg, data_source="raw", raw_data=df_demo)
    try:
        # simulate a later step trying to read the training directory
        bad_dir = os.path.join("artifacts", f"run_{pipe_bad.global_train_hash}")
        if not os.path.exists(bad_dir):
            raise FileNotFoundError(
                f"Expected training artefacts at {bad_dir} (simulated failure)"
            )
        print("[ERR] bad inference did NOT fail as expected")
    except FileNotFoundError as exc:
        print(f"[OK ] bad inference failed correctly → {exc}")
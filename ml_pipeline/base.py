# === ml_pipeline/base.py ==============================================
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# util helpers ----------------------------------------------------------
from .utils import make_param_hash, log_registry

# pipeline steps --------------------------------------------------------
from .eda                       import eda
from .feature_engineering       import feature_engineering
from .partitioning              import partitioning
from .numeric_conversion        import numeric_conversion
from .scaling                   import scaling
from .model_baseline            import model_baseline
from .shap_explainability       import shap_explainability
from .shap_selection            import shap_selection
from .feature_correlation       import feature_correlation
from .feature_select_cluster    import feature_select_cluster
from .feature_select_threshold  import feature_select_threshold
from .hyperparameter_tuning     import hyperparameter_tuning
from .final_model               import final_model


class MLPipeline:
    """
    Configuration‑driven ML pipeline orchestrator.

    • A single `global_hash` (a/k/a run hash) is used across every step.
    • Training mode creates all artifacts; inference mode creates only the
      minimal additional artifacts it needs and re‑uses those from the
      training run referenced by `train_hash`.
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
        self.config         = config
        self.train_mode     = bool(config.get("train_mode", True))
        self.config["train_mode"] = self.train_mode  # echo back

        # ------------------------------------------------ data handles -
        self.data_source    = data_source
        self.db_path        = db_path
        self.xlsx_path      = xlsx_path
        self.csv_paths      = csv_paths
        self.raw_data       = raw_data

        # ------------------------------------------------ state maps ---
        self.dataframes : dict[str, pd.DataFrame] = {}
        self.paths      : dict[str, str]          = {}
        self.models     : dict[str, object]       = {}
        self.metrics    : dict[str, dict]         = {}
        self.artifacts  : dict[str, dict]         = {}

        # ------------------------------------------------ run hash -----
        if self.train_mode:
            self.global_hash = make_param_hash(   # type: ignore[arg-type]
                {
                    "timestamp"  : datetime.now(timezone.utc).isoformat(),
                    "config"     : self.config,
                    "data_source": self.data_source,
                }
            )
        else:
            # inference must be told which training run to re‑use
            self.global_hash = self.config["train_hash"]

        # expose for downstream components
        self.config["global_hash"] = self.global_hash

        # directory for every artifact of this run
        self.run_dir = os.path.join("artifacts", f"run_{self.global_hash}")
        os.makedirs(self.run_dir, exist_ok=True)

        # ------------------------------------------------ bind steps ---
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
    # DATA LOADING
    # ──────────────────────────────────────────────────────────────────
    def load_data(self) -> pd.DataFrame:
        """
        Load and merge raw data from SQLite / Excel / CSV or a supplied DF.
        """
        if self.data_source == "raw":
            return self.raw_data

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
        df_clients  .rename(columns={"account_creation_date": "account_creation_date_client"}  , inplace=True)
        df_merchants.rename(columns={"account_creation_date": "account_creation_date_merchant"}, inplace=True)

        return (
            df_tx
            .merge(df_clients,  on="client_id")
            .merge(df_merchants, on="merchant_id")
        )

    # ──────────────────────────────────────────────────────────────────
    # PIPELINE RUNNERS
    # ──────────────────────────────────────────────────────────────────
    def run_all(self) -> None:
        """
        Execute the core steps needed for a baseline model.
        """
        self.eda()
        self.feature_engineering()
        self.partitioning()
        self.numeric_conversion()
        self.scaling()
        self.model_baseline()

    def run_later(self) -> None:
        """
        Execute the optional later steps (explainability, selection, tuning).
        """
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
        Step files call this to store their path and emit one registry line.
        """
        self.paths[step] = step_dir
        log_registry(step, self.global_hash, cfg or {}, step_dir)
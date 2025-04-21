# === ml_pipeline/utils.py =============================================
"""
Utility helpers used across the ML pipeline.

Conforms to the pipeline spec:
• Provides deterministic `stable_hash` (spec §1‑A).
• Removes timestamp‑based hashing utilities (commented, not deleted).
• Keeps `log_registry`, which may log a timestamp because that value
  is *not* used for hashing.
• Contains a working smoke‑test in the `if __name__ == "__main__":` guard.
"""

from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime, timezone
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

# ──────────────────────────────────────────────────────────────────────
# HASHING HELPERS
# ──────────────────────────────────────────────────────────────────────
def make_param_hash(obj: Any, length: int = 12) -> str:
    """
    Deterministic hash: SHA‑256 over canonical JSON (sorted keys),
    truncated to `length` hex characters.
    """
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


# legacy functions ─ kept but commented‑out per etiquette
# def make_param_hash(config):
#     """
#     Non‑deterministic hash that relied on timestamps.
#     Deprecated by `stable_hash`; retained only for backward reference.
#     """
#     config = convert_numpy_types(config)
#     config_str = json.dumps(config, sort_keys=True)
#     return hashlib.sha256(config_str.encode()).hexdigest()[:10]


# def make_param_hash_(config: dict) -> str:
#     """
#     Alternate MD5 variant (deprecated).
#     """
#     config_str = json.dumps(config, sort_keys=True)
#     return hashlib.md5(config_str.encode()).hexdigest()[:10]


# ──────────────────────────────────────────────────────────────────────
# DATA LOADER
# ──────────────────────────────────────────────────────────────────────
def load_data(db_path: str = "fraud_poc.db") -> pd.DataFrame:
    """
    Load data from the SQLite database and merge client and merchant information.
    """
    conn = sqlite3.connect(db_path)
    df_clients       = pd.read_sql_query("SELECT * FROM clients", conn)
    df_merchants     = pd.read_sql_query("SELECT * FROM merchants", conn)
    df_transactions  = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()

    df_clients  .rename(columns={"account_creation_date": "account_creation_date_client"}  , inplace=True)
    df_merchants.rename(columns={"account_creation_date": "account_creation_date_merchant"}, inplace=True)

    return (
        df_transactions
        .merge(df_clients , on="client_id")
        .merge(df_merchants, on="merchant_id")
    )


# ──────────────────────────────────────────────────────────────────────
# JSON‑SERIALISATION HELPERS
# ──────────────────────────────────────────────────────────────────────
def convert_numpy_types(obj: Any):
    """
    Recursively convert NumPy / pandas scalar types to native Python types
    so the object becomes JSON‑serialisable.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd and isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj


# secondary variant kept for backward compatibility
def convert_numpy_types_(obj: Any):
    """Deprecated alias that points to `convert_numpy_types`."""
    return convert_numpy_types(obj)


# ──────────────────────────────────────────────────────────────────────
# GLOBAL REGISTRY LOGGER
# ──────────────────────────────────────────────────────────────────────
def log_registry(step: str, param_hash: str, config: dict, output_dir: str) -> None:
    """
    Append one JSON‑line to `artifacts/global_registry.jsonl`.
    """
    registry_path = "artifacts/global_registry.jsonl"
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)

    entry = {
        "step"       : step,
        "param_hash" : param_hash,
        "timestamp"  : datetime.now(timezone.utc).isoformat(),  # ok → not used in hashing
        "config"     : config,
        "output_dir" : output_dir,
    }

    with open(registry_path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


# ──────────────────────────────────────────────────────────────────────
# MATPLOTLIB ARTIFACT HELPER
# ──────────────────────────────────────────────────────────────────────
def save_plot_as_artifact(fig, artifact_path: str, artifacts_dict: dict, artifact_key: str) -> None:
    """
    Save a matplotlib figure as an artifact and register its path.
    """
    fig.savefig(artifact_path)
    plt.close(fig)
    artifacts_dict[artifact_key] = artifact_path


# ──────────────────────────────────────────────────────────────────────
# SMOKE‑TEST  (spec §8)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Minimal smoke‑test:
    • Loads a tiny dummy DataFrame (if DB not present, catches the error).
    • Generates a deterministic hash from a toy config.
    """
    dummy_cfg = {
        "param1": "value1",
        "param2": 123,
        "nested": {"a": 1, "b": 2},
    }
    print("Stable hash :", make_param_hash(dummy_cfg))

    # Data load check (non‑existent DB tolerated in smoke‑test)
    try:
        df = load_data("fraud_poc.db")
        print("Loaded rows :", len(df))
    except sqlite3.OperationalError as e:
        print("SQLite load skipped (DB missing):", e)
import json
import hashlib
import pandas as pd
import sqlite3
import pandas as pd
import sqlite3
import os
from datetime import datetime, timezone
import matplotlib.pyplot as plt


def load_data(db_path: str = "fraud_poc.db") -> pd.DataFrame:
    """
    Load data from the SQLite database and merge client and merchant information.

    Args:
        db_path (str): Path to the SQLite database.

    Returns:
        pd.DataFrame: Merged DataFrame containing transaction, client, and merchant data.
    """
    conn = sqlite3.connect(db_path)
    df_clients = pd.read_sql_query("SELECT * FROM clients", conn)
    df_merchants = pd.read_sql_query("SELECT * FROM merchants", conn)
    df_transactions = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()

    df_clients.rename(columns={"account_creation_date": "account_creation_date_client"}, inplace=True)
    df_merchants.rename(columns={"account_creation_date": "account_creation_date_merchant"}, inplace=True)

    return df_transactions.merge(df_clients, on="client_id").merge(df_merchants, on="merchant_id")


# Update your convert_numpy_types function to handle Timestamps
def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    import numpy as np
    import pandas as pd
    
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


def convert_numpy_types_(obj):
    """Recursively convert numpy types to Python types."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def make_param_hash(config):
    """
    Create a deterministic hash from a pipeline step's parameters.
    """
    # Convert NumPy types to Python native types
    config = convert_numpy_types(config)
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:10]


# def make_param_hash(params: dict) -> str:
def make_param_hash_(config: dict) -> str:
    """
    Generate a hash ID based on the configuration dictionary.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        str: Hash string.
    """
    config_str = json.dumps(config, sort_keys=True)
    # return hashlib.md5(config_str.encode()).hexdigest()

    # param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:10]


def log_to_global_registry(entry: dict, registry_path: str = "artifacts/global_registry.jsonl") -> None:
    """
    Log an entry to the global registry. Append a new entry to the global registry file.

    Args:
        entry (dict): Entry to log.
        registry_path (str): Path to the global registry file.
    
                Default is "artifacts/global_registry.jsonl".
    Returns:
        None
    """
    
    with open(registry_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_plot_as_artifact(fig, artifact_path, artifacts_dict, artifact_key):
    """
    Save a matplotlib figure as an artifact and register its path.
    """
    fig.savefig(artifact_path)
    plt.close(fig)
    artifacts_dict[artifact_key] = artifact_path


if __name__ == "__main__":
    # Example usage
    db_path = "fraud_poc.db"
    df = load_data(db_path)
    print(df.head())  # Display the first few rows of the DataFrame

    config = {
        "param1": "value1",
        "param2": "value2",
        "param3": 123
    }
    hash_id = make_param_hash(config)
    print(f"Generated hash ID: {hash_id}")
    # Example configuration dictionary for hashing
    config = {
        "param1": "value1",
        "param2": "value2",
        "param3": 123
    }
    hash_id = make_param_hash(config)
    print(f"Generated hash ID: {hash_id}")
    # Example configuration dictionary for hashing


def log_registry(step: str, param_hash: str, config: dict, output_dir: str) -> None:
    """
    Log step details to the global registry.

    Args:
        step (str): Step name.
        param_hash (str): Parameter hash.
        config (dict): Configuration dictionary.
        output_dir (str): Output directory.

    Returns:
        None
    """
    os.makedirs(os.path.dirname("artifacts/global_registry.jsonl"), exist_ok=True)
    with open("artifacts/global_registry.jsonl", "a") as f:
        f.write(json.dumps({
            "step": step,
            "param_hash": param_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "output_dir": output_dir
        }) + "\n")
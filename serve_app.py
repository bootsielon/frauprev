import streamlit as st
import pandas as pd
import os
from ml_pipeline.base import MLPipeline

# --- CONFIGURATION ---
ARTIFACTS_ROOT = "artifacts"

def list_models(artifacts_root):
    """Find all model_baseline_* directories and list model files within them that start with 'model_'."""
    model_files = []
    for d in os.listdir(artifacts_root):
        dir_path = os.path.join(artifacts_root, d)
        if os.path.isdir(dir_path) and d.startswith("model_baseline"):
            for f in os.listdir(dir_path):
                if (f.startswith('model_') and (f.endswith('.json') or f.endswith('.pkl'))):
                    model_files.append((d, f))
    return model_files

def main():
    st.title("Fraud Prediction Service")

    # 1. Model selection
    st.header("1. Select Model")
    model_files = list_models(ARTIFACTS_ROOT)
    if not model_files:
        st.error("No models found in artifacts directory.")
        return
    model_choice = st.selectbox("Choose a trained model:", [f"{d}/{f}" for d, f in model_files])
    selected_dir, model_file = model_choice.split("/", 1)
    model_hash = selected_dir.split("_", 2)[-1]  # Adjust as needed

    # 2. File upload
    st.header("2. Upload Data File")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file with transactions", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            # Read all sheets
            xls = pd.ExcelFile(uploaded_file)
            try:
                df_clients = pd.read_excel(xls, sheet_name="clients")
                df_merchants = pd.read_excel(xls, sheet_name="merchants")
                df_transactions = pd.read_excel(xls, sheet_name="transactions")
            except Exception as e:
                st.error(f"Error reading sheets: {e}")
                return

            # Merge as in your pipeline's load_data
            df = df_transactions.merge(df_clients, on="client_id", how="left") \
                                .merge(df_merchants, on="merchant_id", how="left")
        st.write("Preview of merged data:", df.head())

        # 3. Run pipeline in inference mode
        pipeline = MLPipeline(
            config={
                "train_mode": False,
                "data_source": "raw",
                "use_mlflow": True,
                "train_hash": model_hash
            },  # Add any other config needed
            raw_data=df
        )
        pipeline.run_all()

        # 4. Show and download results
        scored_df = pipeline.dataframes.get("scored", pipeline.dataframes.get("test", df))
        st.write("Results preview:", scored_df.head())
        st.header("4. Download Results")
        out_csv = scored_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions as CSV",
            data=out_csv,
            file_name="scored_transactions.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
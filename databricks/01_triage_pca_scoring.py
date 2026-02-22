import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================================================================
# TRIAGE - LAYER 1: DATA INGESTION & PCA SEVERITY SCORING (DATABRICKS NOTEBOOK)
# ==============================================================================
# INSTRUCTIONS FOR DATABRICKS:
# 1. Upload your 6 raw CSV files to your Databricks Workspace (DBFS).
# 2. Update the `base_path` in the `load_and_clean_data` function below.
# 3. Run this notebook to generate the `triage_master_scores.csv`
# ==============================================================================

# Dictionary to map some common country names to ISO3 for UCDP conflict data
ISO3_MAP = {
    'Afghanistan': 'AFG',
    'Angola': 'AGO',
    'Burundi': 'BDI',
    'Sudan': 'SDN',
    'Haiti': 'HTI',
    'Ukraine': 'UKR',
    'Syria': 'SYR',
    'Yemen': 'YEM'
# Add more mappings as necessary for the full dataset
}

# Check if 'spark' exists in the global namespace (which it does in Databricks notebooks)
in_databricks = 'spark' in globals() or 'spark' in locals()

def load_and_clean_data(base_path="../assets"):
    """
    Loads the real Hacklytics UN datasets.
    If running inside Databricks, it uses PySpark to read directly from the Unity Catalog (workspace.raw).
    If running locally, it falls back to reading the CSV files.
    """
    print("Loading datasets...")
    
    # 1. FTS Funding Data
    try:
        if in_databricks:
            df_fts_spark = spark.table("workspace.raw.fts_requirements_funding_global")
            df_fts = df_fts_spark.toPandas()
        else:
            path_fts = os.path.join(base_path, "fts_requirements_funding_global.csv")
            df_fts = pd.read_csv(path_fts)
            
        # VERY IMPORTANT: The UN FTS data contains a second row of metadata tags (e.g. #country+code, #date+year, #value+funding)
        # We MUST drop this string row before doing math, otherwise Pandas converts the entire column to NaN.
        df_fts = df_fts[~df_fts['year'].astype(str).str.contains('#', na=False)]
            
        # Flexible renaming for year, requirements, funding, countrycode
        rename_map = {}
        for col in df_fts.columns:
            lower_col = col.lower()
            if lower_col in ['countrycode', 'countrycode3', 'iso3']: rename_map[col] = 'iso3'
            if lower_col == 'requirements': rename_map[col] = 'funding_required'
            if lower_col == 'funding': rename_map[col] = 'funding_received'
            if lower_col == 'year': rename_map[col] = 'year'
            
        df_fts.rename(columns=rename_map, inplace=True)
        
        if 'year' in df_fts.columns:
            latest_year = df_fts['year'].max()
            df_fts = df_fts[df_fts['year'] == latest_year]
        
        keep_cols = ['iso3']
        if 'funding_required' in df_fts.columns: 
            # Clean string numbers (e.g. "1,000.00") if necessary
            if df_fts['funding_required'].dtype == 'object':
                df_fts['funding_required'] = df_fts['funding_required'].astype(str).str.replace(',', '').str.replace('$', '')
            df_fts['funding_required'] = pd.to_numeric(df_fts['funding_required'], errors='coerce')
            keep_cols.append('funding_required')
            
        if 'funding_received' in df_fts.columns: 
            if df_fts['funding_received'].dtype == 'object':
                df_fts['funding_received'] = df_fts['funding_received'].astype(str).str.replace(',', '').str.replace('$', '')
            df_fts['funding_received'] = pd.to_numeric(df_fts['funding_received'], errors='coerce')
            keep_cols.append('funding_received')
            
        df_fts = df_fts[keep_cols]
    except Exception as e:
        print(f"Warning: FTS Data failed to load ({e}). Creating mock structure.")
        df_fts = pd.DataFrame({'iso3': pd.Series(dtype='str'), 'funding_required': pd.Series(dtype='float'), 'funding_received': pd.Series(dtype='float')})
        
    # 2. IPC Food Security Data
    try:
        if in_databricks:
            df_ipc_spark = spark.table("workspace.raw.ipc_global_national_wide_latest")
            df_ipc = df_ipc_spark.toPandas()
        else:
            path_ipc = os.path.join(base_path, "ipc_global_national_wide_latest.csv")
            df_ipc = pd.read_csv(path_ipc)
            
        rename_map = {}
        for col in df_ipc.columns:
            if col.lower() == 'country': rename_map[col] = 'iso3'
            if col.lower() == 'phase 3+ number current': rename_map[col] = 'ipc_phase_3_plus'
            
        df_ipc.rename(columns=rename_map, inplace=True)
        
        if df_ipc['ipc_phase_3_plus'].dtype == 'object':
            df_ipc['ipc_phase_3_plus'] = df_ipc['ipc_phase_3_plus'].astype(str).str.replace(',', '')
        df_ipc['ipc_phase_3_plus'] = pd.to_numeric(df_ipc['ipc_phase_3_plus'], errors='coerce')
        df_ipc = df_ipc.groupby('iso3')['ipc_phase_3_plus'].sum().reset_index()
    except Exception as e:
        print(f"Warning: IPC Data failed to load ({e}). Creating mock structure.")
        df_ipc = pd.DataFrame({'iso3': pd.Series(dtype='str'), 'ipc_phase_3_plus': pd.Series(dtype='float')})

    # 3. UCDP Conflict Data
    try:
        if in_databricks:
            df_conflict_spark = spark.table("workspace.raw.dataset_3_ucdp_ged")
            df_conflict = df_conflict_spark.toPandas()
        else:
            path_conflict = os.path.join(base_path, "Dataset3-ucdp_ged.csv")
            df_conflict = pd.read_csv(path_conflict, engine='python', on_bad_lines='skip')
            
        rename_map = {}
        for col in df_conflict.columns:
            if col.lower() == 'country': rename_map[col] = 'country'
            if col.lower() == 'best': rename_map[col] = 'fatalities'
            
        df_conflict.rename(columns=rename_map, inplace=True)
        
        if df_conflict['fatalities'].dtype == 'object':
             df_conflict['fatalities'] = df_conflict['fatalities'].astype(str).str.replace(',', '')
        df_conflict['fatalities'] = pd.to_numeric(df_conflict['fatalities'], errors='coerce')
        df_conflict = df_conflict.groupby('country')['fatalities'].sum().reset_index()
        
        # Map country names to ISO3 for merging
        df_conflict['iso3'] = df_conflict['country'].map(ISO3_MAP)
        df_conflict = df_conflict.dropna(subset=['iso3'])
        df_conflict = df_conflict[['iso3', 'fatalities']]
    except Exception as e:
        print(f"Warning: UCDP Data failed to load ({e}). Creating mock structure.")
        df_conflict = pd.DataFrame({'iso3': pd.Series(dtype='str'), 'fatalities': pd.Series(dtype='float')})

    # Merge all available datasets on 'iso3'
    print("Merging datasets on ISO3...")
    df_master = df_fts.merge(df_ipc, on='iso3', how='outer')\
                      .merge(df_conflict, on='iso3', how='outer')
    
    # Ensure all target columns exist and are explicitly float type so PySpark schema inference succeeds
    expected_numeric_cols = ['funding_required', 'funding_received', 'ipc_phase_3_plus', 'fatalities']
    for col in expected_numeric_cols:
        if col not in df_master.columns:
            df_master[col] = 0.0
        else:
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce').fillna(0.0)
            
    # Filter out empty ISO3s
    df_master = df_master[df_master['iso3'] != 0]
    df_master = df_master.dropna(subset=['iso3'])
    return df_master


def calculate_pca_severity(df):
    """
    Uses Unsupervised Machine Learning (PCA) to calculate the Severity Score.
    Based on fatalities, phase 3+ population, and a normalized funding gap proxy.
    """
    print("Running PCA to calculate Severity Scores...")
    
    # We will use the available data columns as our features
    features = []
    if 'fatalities' in df.columns: features.append('fatalities')
    if 'ipc_phase_3_plus' in df.columns: features.append('ipc_phase_3_plus')
    
    if not features:
        print("No features available for PCA. Returning zeros.")
        df['Crisis_Severity_Score'] = 0
        return df

    X = df[features]
    
    # Standardize the data (Z-scores)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA(n_components=1)
    severity_component = pca.fit_transform(X_scaled)
    
    # Normalize the PCA output to a 0-100 scale
    min_val = severity_component.min()
    max_val = severity_component.max()
    
    if max_val == min_val:
        severity_score_100 = np.zeros(len(severity_component))
    else:
        severity_score_100 = 100 * (severity_component - min_val) / (max_val - min_val)
    
    df['Crisis_Severity_Score'] = np.round(severity_score_100, 2)
    return df


def calculate_funding_gap_and_flags(df):
    """
    Calculates the Funding Coverage Ratio and identifies the Red Zones.
    """
    print("Calculating Funding Gaps and identifying Red Zones...")
    
    # Compute Funding Coverage Ratio
    if 'funding_required' in df.columns and 'funding_received' in df.columns:
        df['Funding_Coverage_Ratio'] = np.where(
            df['funding_required'] > 0,
            df['funding_received'] / df['funding_required'],
            1.0
        )
    else:
        df['Funding_Coverage_Ratio'] = 0.0

    df['Funding_Coverage_Ratio'] = np.round(df['Funding_Coverage_Ratio'], 3)
    
    # THE RED ZONE LOGIC: High Severity (>75) AND Low Coverage (<0.30)
    df['Is_Red_Zone'] = np.where(
        (df['Crisis_Severity_Score'] > 75) & (df['Funding_Coverage_Ratio'] < 0.30),
        True, 
        False
    )
    return df


if __name__ == "__main__":
    # Pointing base_path to the local assets folder just for local testing / demo
    # IN DATABRICKS: change to "/dbfs/FileStore/tables"
    local_asset_path = "../assets"
    
    # Ensure local path is used only dynamically, otherwise fallback to mock
    if os.path.exists(os.path.join(local_asset_path, "fts_requirements_funding_global.csv")):
        df_raw = load_and_clean_data(base_path=local_asset_path)
    else:
        # Absolute path for current hackathon environment
        abs_path = "/Users/hidhayathnishamohamedidris/Hacklytics/assets"
        df_raw = load_and_clean_data(base_path=abs_path)

    if not df_raw.empty:
        # Run the Machine Learning (PCA) Math
        df_scored = calculate_pca_severity(df_raw)
        
        # Calculate Financial Gap and Red Zones
        df_final = calculate_funding_gap_and_flags(df_scored)
        
        # Display the final output
        print("\n================ FINAL TRIAGE SCORES ================")
        print(df_final[['iso3', 'Crisis_Severity_Score', 'Funding_Coverage_Ratio', 'Is_Red_Zone']].sort_values(by='Crisis_Severity_Score', ascending=False).head(15))
        
        # Output save path
        try:
            if in_databricks:
                # Save as a Delta Table in Unity Catalog
                spark_df = spark.createDataFrame(df_final)
                spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.raw.triage_master_scores")
                print("\nSaved scoring results to Unity Catalog: workspace.raw.triage_master_scores")
            else:
                output_csv = os.path.join(local_asset_path, "triage_master_scores.csv")
                df_final.to_csv(output_csv, index=False)
                print(f"\nSaved scoring results locally to: {output_csv}")
        except Exception as e:
            print(f"Could not save output file: {e}")
    else:
        print("Data processing resulted in an empty dataframe.")

# Cleaning script so that all data can be cleaned
import pandas as pd
import numpy as np
import re

def clean_accounts_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # : Normalize column names
    df.columns = [c.strip().lower().replace(' ', '_').replace('.', '').replace('-', '_') for c in df.columns]
    
    # : Trim whitespace and normalize placeholders to NaN
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace({'nan': np.nan, '': np.nan, 'N/A': np.nan, 'NA': np.nan, 'Unknown': np.nan, 'none': np.nan})
    
    # : Parse activated_date to datetime before any other operations
    if 'activated_date' in df.columns:
        df['activated_date'] = pd.to_datetime(df['activated_date'], errors='coerce')
        # Fill missing dates with the median date
        if df['activated_date'].isna().any():
            median_date = df['activated_date'].median()
            if pd.notna(median_date):
                df['activated_date'] = df['activated_date'].fillna(median_date)
    
    # : Extract MCC numeric code (support both 'mcc' or 'mcc')
    if 'mcc' in df.columns:
        df['mcc_code'] = df['mcc'].astype(str).str.extract(r'(\d{3,4})')[0]
        df['mcc_code'] = pd.to_numeric(df['mcc_code'], errors='coerce')
    
    # : Fill missing values - categorical with mode, numeric with mean
    for col in df.columns:
        if df[col].dtype == 'object':
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna('Unknown')
        elif df[col].dtype in ['float64', 'int64']:
            if df[col].isna().any():
                mean_val = df[col].mean()
                if pd.notna(mean_val):
                    df[col] = df[col].fillna(mean_val)
                    
                    if 'account_status' in df.columns:
                        df['conversion'] = df['account_status'].apply(
                            lambda x: 1 if str(x).strip().lower() == 'activated' else 0
                        )
                    else:
                        df['conversion'] = np.nan


    
    # : Standardize selected categorical fields to Title Case
    for col in ['country', 'referral_source', 'account_type', 'account_status', 'sector']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: x.title() if pd.notna(x) else x)
    
    # : Derive date features from activated_date
    if 'activated_date' in df.columns:
        df['activate_year'] = df['activated_date'].dt.year
        df['activate_month'] = df['activated_date'].dt.month
    else:
        df['activate_year'] = np.nan
        df['activate_month'] = np.nan
    
    return df

#There are no duplicate rows according to the data profile obtained

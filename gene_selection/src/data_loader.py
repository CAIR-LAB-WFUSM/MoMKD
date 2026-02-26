# src/data_loader.py
import pandas as pd
from pathlib import Path

def make_columns_unique(columns):

    counts = {}
    new_columns = []
    for col in columns:
        if col in counts:
            counts[col] += 1
            new_columns.append(f"{col}.{counts[col]}")
        else:
            counts[col] = 0
            new_columns.append(col)
    return new_columns

def load_and_merge_data(gene_path: Path, meta_path: Path) -> pd.DataFrame:

    print("1. Loading and merging data...")
    
 
    df_gene = pd.read_csv(gene_path, index_col=0).T
    

    if df_gene.columns.has_duplicates:
        print("   - WARNING: Duplicate gene names found. Making columns unique.")
        unique_cols = make_columns_unique(df_gene.columns)
        df_gene.columns = unique_cols


    df_gene.index.name = "case_id"
    
  
    df_meta = pd.read_csv(meta_path)

    df_meta['case_id'] = df_meta['slide_id'].str[:15]
    
    id_map = df_meta.drop_duplicates(subset=['case_id']).set_index('case_id')['slide_id']

    df_meta_subset = df_meta[['case_id', 'pr_status_by_ihc']].copy()
    

    merged_df = pd.merge(df_gene, df_meta_subset, on='case_id', how='inner')

    merged_df = merged_df.drop_duplicates(subset=['case_id'])
    merged_df = merged_df.set_index('case_id')
    
    print(f"   - Merged data shape: {merged_df.shape}")
    return merged_df,id_map
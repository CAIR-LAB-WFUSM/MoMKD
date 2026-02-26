# src/export.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from sklearn.metrics import classification_report, roc_auc_score

from sklearn.linear_model import LogisticRegression

def evaluate_and_save_report(X_train, y_train, X_test, y_test, final_genes, report_path, random_state):
    
    print("4. Evaluating final selected features on test set...")
    if not final_genes:
        print("   - No features selected. Skipping evaluation.")
        with open(report_path, 'w') as f:
            f.write("No genes were selected, so no classification report could be generated.\n")
        return


    eval_model = LogisticRegression(penalty='l2', random_state=random_state, solver='liblinear')
    eval_model.fit(X_train[final_genes], y_train)

    y_pred = eval_model.predict(X_test[final_genes])

    y_pred_proba = eval_model.predict_proba(X_test[final_genes])[:, 1]

    report = classification_report(y_test, y_pred, target_names=['odxL', 'odxH'])

    auc_score = roc_auc_score(y_test, y_pred_proba)
    # --------------------------
    
    print("   - Classification Report:")
    print(report)

    print(f"   - AUC Score: {auc_score:.4f}")
    # --------------------------
    

    with open(report_path, 'w') as f:
        f.write("Classification Report for Selected Features\n")
        f.write("="*50 + "\n")
        f.write(report)
        f.write("\n" + "="*50 + "\n")

        f.write(f"AUC Score: {auc_score:.4f}\n")
        # -------------------------------
        
    print(f"   - Report saved to {report_path}")


def export_results(final_genes: List[str], X_scaled_full: pd.DataFrame, id_map: pd.Series, npy_dir: Path, gene_list_path: Path):

    print("5. Exporting final results...")
    if not final_genes:
        print("   - No features to export.")
        return
        

    pd.DataFrame({'gene': final_genes}).to_csv(gene_list_path, index=False)
    print(f"   - Final gene list saved to {gene_list_path}")
    

    npy_dir.mkdir(parents=True, exist_ok=True)
    
    X_final_subset = X_scaled_full[final_genes]
    
    for case_id, gene_vector in X_final_subset.iterrows():
        if case_id in id_map.index:
            full_slide_id = id_map.loc[case_id]
            filename_base = full_slide_id.split('.')[0]
            output_path = npy_dir / f"{filename_base}.npy"
            np.save(output_path, gene_vector.values)
        else:
            print(f"   - WARNING: case_id {case_id} not found in id_map. Skipping .npy export for this sample.")
    
    print(f"   - Saved {len(X_final_subset)} .npy files to {npy_dir}")
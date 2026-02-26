# main.py
import argparse
from pathlib import Path

from src.data_loader import load_and_merge_data
from src.preprocessing import preprocess_data
from src.export import evaluate_and_save_report, export_results
from src.feature_selection import get_top_n_genes_by_xgboost
def main(args):

    gene_path = Path(args.gene_expression_path)
    meta_path = Path(args.metadata_path)
    results_dir = Path(args.results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    

    merged_df ,id_map = load_and_merge_data(gene_path=gene_path, meta_path=meta_path)
    

    X_train, X_test, y_train, y_test, X_scaled_full  = preprocess_data(
        df=merged_df,
        target_col=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state
    )

    final_selected_genes = get_top_n_genes_by_xgboost(
        X_train=X_train,
        y_train=y_train,
        n_genes_to_select=args.n_genes,
        random_state=args.random_state
    )
    

    report_path = results_dir / "classification_report.txt"
    gene_list_path = results_dir / "her2_ensemble_selected_genes.csv"
    npy_output_dir = results_dir / "selected_genes_npy"
    
 
    evaluate_and_save_report(
        X_train, y_train, X_test, y_test, 
        final_genes=final_selected_genes,
        report_path=report_path,
        random_state=args.random_state
    )
    

    export_results(
        final_genes=final_selected_genes,
        X_scaled_full=X_scaled_full,
        npy_dir=npy_output_dir,
        id_map=id_map,
        gene_list_path=gene_list_path
    )
    
    print(f"\nPipeline finished successfully! Results are in '{results_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HER2 Gene Selection Pipeline using Ensemble Methods.")

    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--target-column', type=str, default='HER2Calc', help='Name of the target column in the metadata file.')
    

    parser.add_argument('--gene-expression-path', type=str, required=True, help='Path to the gene expression CSV file (TCGA_gene.csv).')
    parser.add_argument('--metadata-path', type=str, required=True, help='Path to the metadata CSV file (BRCA_processed_with_paths.csv).')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save all output files.')

    parser.add_argument('--n-genes', type=int, default=768, help='Number of top genes to select based on importance.')
    args = parser.parse_args()
    main(args)
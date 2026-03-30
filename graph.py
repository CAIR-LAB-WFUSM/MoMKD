import h5py
import numpy as np
from sklearn.neighbors import kneighbors_graph
import os
from tqdm import tqdm
import argparse
import shutil

def process_single_h5(source_path, dest_path, k_neighbors=8, force_recompute=False):
    """
    Reads an H5 file, computes the graph, and saves a new H5 file with the graph included.
    
    Args:
        source_path (str): Path to the original H5 feature file.
        dest_path (str): Path to save the new H5 file.
        k_neighbors (int): Number of neighbors for the KNN graph.
        force_recompute (bool): If True, recomputes the graph even if the destination file exists.
    """
    if os.path.exists(dest_path) and not force_recompute:
        return "skipped_exist"

    try:
        # Copy the original file to the new location first
        shutil.copyfile(source_path, dest_path)
        
        # Now, open the NEW file in read-write mode to add the graph
        with h5py.File(dest_path, 'a') as hf:
            edge_index_key = f'edge_index_k{k_neighbors}'
            
            if 'coords' not in hf:
                # print(f"Warning: 'coords' not found in {os.path.basename(source_path)}. Graph not added.")
                return "no_coords"
            
            coords = hf['coords'][:]
            
            if coords.shape[0] <= k_neighbors:
                # print(f"Warning: Not enough patches ({coords.shape[0]}) in {os.path.basename(source_path)}.")
                return "not_enough_points"
                
            adj_matrix = kneighbors_graph(coords, k_neighbors, mode='connectivity', include_self=False, n_jobs=-1)
            edge_index = np.stack(adj_matrix.nonzero())
            
            # If the key happens to be in the copied file (e.g., from a partial run)
            if edge_index_key in hf:
                del hf[edge_index_key]
                
            hf.create_dataset(edge_index_key, data=edge_index, compression="gzip")
        return "success"

    except Exception as e:
        print(f"ERROR processing file {source_path}: {e}")
        # Clean up the partially created file if an error occurs
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return "error"

def main(args):
    source_dir = args.source_dir
    dest_dir = args.dest_dir
    k_val = args.k_neighbors
    
    # --- Safety Checks ---
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at {source_dir}")
        return
        
    if source_dir == dest_dir:
        print("Error: Source and destination directories cannot be the same. Please specify a different destination.")
        return
        
    os.makedirs(dest_dir, exist_ok=True)
        
    h5_files = [f for f in os.listdir(source_dir) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"No .h5 files found in {source_dir}")
        return
        
    print(f"Source Directory: {source_dir}")
    print(f"Destination Directory: {dest_dir}")
    print(f"Found {len(h5_files)} H5 files to process.")
    print(f"Graph settings: k_neighbors = {k_val}")
    if args.force_recompute:
        print("Force recompute is ON. Existing files in destination will be overwritten.")

    pbar = tqdm(h5_files)
    results = {"success": 0, "skipped_exist": 0, "no_coords": 0, "not_enough_points": 0, "error": 0}

    for filename in pbar:
        pbar.set_description(f"Processing {filename}")
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        
        status = process_single_h5(source_path, dest_path, k_neighbors=k_val, force_recompute=args.force_recompute)
        results[status] += 1
        
    print("\n--- Preprocessing Complete ---")
    print(f"Successfully created/updated: {results['success']}")
    print(f"Skipped (already exists in dest): {results['skipped_exist']}")
    print(f"Skipped (no coordinates found): {results['no_coords']}")
    print(f"Skipped (not enough patches): {results['not_enough_points']}")
    print(f"Errors encountered: {results['error']}")
    print("----------------------------")
    print(f"New H5 files with graphs are saved in: {dest_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-compute KNN graphs and save to new H5 files.')
    
    parser.add_argument('--source_dir', type=str, 
                        default="/isilon/datalake/gurcan_rsch/scratch/WSI/yongxin/foundation/trident/tcga_all_uni2/20x_896px_0px_overlap/features_uni_v2",
                        help='Directory containing the ORIGINAL H5 feature files.')
                        
    parser.add_argument('--dest_dir', type=str, 
                        default="/isilon/datalake/gurcan_rsch/scratch/WSI/yongxin/foundation/trident/tcga_all_uni2/20x_896px_0px_overlap/graph_feature_bcnb",
                        help='Directory to save the NEW H5 files with graph information.')
                        
    parser.add_argument('--k_neighbors', type=int, default=8,
                        help='Number of nearest neighbors for graph construction.')
                        
    parser.add_argument('--force_recompute', action='store_true',
                        help='If specified, overwrites existing files in the destination directory.')
                        
    args = parser.parse_args()
    main(args)

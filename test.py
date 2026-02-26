import argparse
import os
import numpy as np
import pandas as pd
import torch
import sys
import h5py
from sklearn.metrics import (roc_auc_score, confusion_matrix, f1_score, 
                             accuracy_score, precision_recall_curve, 
                             average_precision_score)
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import kneighbors_graph


from models.mkd import MKD
import warnings

class Graph_Inference_WSI_Dataset(Dataset):
    def __init__(self, test_df, data_dir, k_neighbors=8):
        self.data_df = test_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.k_neighbors = k_neighbors
    
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        slide_id, label = row['slide_id'], row['label']
        h5_path = os.path.join(self.data_dir, f"{slide_id.replace('.svs', '')}.h5")
        
        try:
            with h5py.File(h5_path, 'r') as hf:
                wsi_features = torch.from_numpy(hf['features'][:]).float()
                edge_index_key = f'edge_index_k{self.k_neighbors}'
                
                if edge_index_key in hf:
                    edge_index = torch.from_numpy(hf[edge_index_key][:]).long()
                
                elif 'coords' in hf:
                    coords = hf['coords'][:]
                    P = coords.shape[0]
                    
                    if P < 2:
                   
                        warnings.warn(
                            f"Slide ID: {slide_id} has only {P} node(s). "
                            f"Cannot form edges. Returning an empty edge_index.",
                            UserWarning
                        )
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                    else:
                        k_eff = min(self.k_neighbors, max(P - 1, 1))
                        adj_matrix = kneighbors_graph(coords, k_eff, mode='connectivity', include_self=False)
                        rows, cols = adj_matrix.nonzero()
                        edge_index = torch.from_numpy(np.stack([rows, cols], axis=0)).long()
                
                else:
                    
                    warnings.warn(
                        f"CRITICAL DATA ISSUE: Slide ID: {slide_id} has no pre-computed '{edge_index_key}' "
                        f"and no 'coords' to compute them from. Returning an empty edge_index.",
                        UserWarning
                    )
                    edge_index = torch.empty((2, 0), dtype=torch.long)

        except Exception as e:
            print(f"ERROR loading {slide_id}: {e}")
            return (torch.zeros(1, 1536), torch.empty((2, 0), dtype=torch.long)), -1, "LOAD_ERROR"
            
        return (wsi_features, edge_index), label, slide_id

class Inference_WSI_Dataset(Dataset):
    def __init__(self, df, wsi_feature_dir):
        self.df = df
        self.wsi_feature_dir = wsi_feature_dir
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id, label = row['slide_id'], row['label']
        feature_path = os.path.join(self.wsi_feature_dir, f"{slide_id.replace('.svs', '')}.h5")
        try:
            with h5py.File(feature_path, 'r') as hf:
                wsi_features = torch.from_numpy(hf['features'][:])
            return wsi_features, label, slide_id
        except Exception as e:
            print(f"ERROR loading {slide_id}: {e}")
            return torch.zeros(1, 1536), -1, "LOAD_ERROR"


def find_best_threshold_f1(labels, probs_pos):
    if len(np.unique(labels)) < 2:
        print("[WARNING] Only one class present for threshold finding. Using default 0.5.")
        return 0.5
    precision, recall, thresholds = precision_recall_curve(np.asarray(labels), np.asarray(probs_pos))
    f1_scores = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1], 
                          out=np.zeros_like(precision[:-1]), where=(precision[:-1] + recall[:-1]) != 0)
    return thresholds[np.argmax(f1_scores)] if len(f1_scores) > 0 else 0.5

def calculate_sota_metrics(labels, preds, probs_pos):
    metrics = {}
    labels, preds, probs_pos = np.array(labels), np.array(preds), np.array(probs_pos)
    
    # Threshold-dependent metrics
    metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
    metrics['accuracy'] = accuracy_score(labels, preds)
    
    # Threshold-independent metrics
    if len(np.unique(labels)) == 2:
        metrics['auc'] = roc_auc_score(labels, probs_pos)
        metrics['auprc'] = average_precision_score(labels, probs_pos)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        metrics.update({'auc': np.nan, 'auprc': np.nan, 'sensitivity': np.nan, 'specificity': np.nan})
    return metrics

def find_full_exp_dir(results_dir, split_dir_name, seed):
    try:
        base_path = os.path.join(results_dir, "5foldcv")
        subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        if not subdirs: return None
        param_code_dir = os.path.join(base_path, subdirs[0])
        for exp_dir_name in os.listdir(param_code_dir):
            if exp_dir_name.endswith(f"_s{seed}"):
                return os.path.join(param_code_dir, exp_dir_name)
        return None
    except Exception: return None

def run_inference(model, df_all_cases, slide_ids, wsi_feature_dir, args):

    df = df_all_cases[df_all_cases['slide_id'].isin(slide_ids)]
    if df.empty: return pd.DataFrame()
    
    if args.model_name in ['gcad_net', 'mkd']:
        dataset = Graph_Inference_WSI_Dataset(df, wsi_feature_dir, 8)
    else: dataset = Inference_WSI_Dataset(df, wsi_feature_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    probs, labels, s_ids, argmax_preds = [], [], [], []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for data_batch, label, s_id in loader:
            if label.item() == -1: continue
            if args.model_name in ['mkd']:
                feats, edges = data_batch[0][0].to(device), data_batch[1][0].to(device)
                if feats.dim() < 2 or edges.numel() == 0: continue
                out = model.forward_inference((feats, edges)) if args.model_name == 'mkd' else model((feats, edges))
                logits = out[0] if isinstance(out, tuple) else out
            else:
                feats = data_batch[0].to(device)
                if feats.dim() < 2: continue
                logits = model(feats.squeeze(0))
        
            Y_hat_argmax = torch.argmax(logits, dim=1).item()
            
            probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
            labels.append(label.item())
            s_ids.append(s_id[0])
            argmax_preds.append(Y_hat_argmax)

    if not labels: return pd.DataFrame()
    probs = np.array(probs)
    return pd.DataFrame({
        'slide_id': s_ids, 
        'label': labels, 
        'p_0': probs[:, 0], 
        'p_1': probs[:, 1],
        'Y_hat_argmax': argmax_preds 
    })


def main(args):
    full_exp_dir = find_full_exp_dir(args.results_dir, args.split_dir_name, args.seed)
    if not full_exp_dir: sys.exit(f"FATAL: Experiment directory for seed {args.seed} not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_fold_metrics = []
    
    df_all_cases = pd.read_csv(args.csv_path, dtype={'slide_id': str})
    df_all_cases['label'] = (df_all_cases[args.label_col].astype(str) == str(args.positive_label)).astype(int)

    for i in range(args.k):
        print(f"\n====================== FOLD {i} ======================")
        split_file = os.path.join(args.split_dir, f'splits_{i}.csv')
        if not os.path.exists(split_file):
            print(f"[WARNING] Split file {split_file} not found. Skipping fold.")
            continue
        splits_df = pd.read_csv(split_file)
        splits_df['test'] = splits_df['test'].astype(str) 
        
        # --- 1. Load Model ---

        if args.model_name == 'mkd': ckpt_filename = f"s_{i}_mkd_checkpoint.pt"
        else: ckpt_filename = f"s_{i}_checkpoint.pt"
        ckpt_path = os.path.join(full_exp_dir, ckpt_filename)
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] Checkpoint {ckpt_path} not found. Skipping fold.")
            continue
            

        if args.model_name == 'mkd': model = MKD(rna_input_dim=args.omic_input_dim, wsi_input_dim=1536, wsi_encoder_dim=args.wsi_encoder_dim,rna_embedding_dim=args.rna_embedding_dim, shared_embedding_dim=args.shared_embedding_dim,num_memory_per_class=args.num_memory, cls_hidden_dim=args.cls_hidden_dim).to(device)
        else: raise ValueError(f"Unknown model: {args.model_name}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # --- 2. Process Validation Set ---
        print(f"  Processing VALIDATION set for fold {i}...")
        val_preds_df = run_inference(model, df_all_cases, splits_df['val'].dropna(), args.data_root_dir, args)
        if val_preds_df.empty:
            print(f"[WARNING] Validation set for fold {i} is empty. Skipping fold.")
            continue
        val_preds_df.to_csv(os.path.join(full_exp_dir, f"val_preds_fold_{i}.csv"), index=False)
        
        best_thr = find_best_threshold_f1(val_preds_df['label'], val_preds_df['p_1'])
        print(f"  Best threshold from validation set (max F1): {best_thr:.4f}")

 
        print(f"  Processing TEST set for fold {i}...")
        test_preds_df = run_inference(model, df_all_cases, splits_df['test'].dropna(), args.data_root_dir, args)
        if test_preds_df.empty:
            print(f"[WARNING] Test set for fold {i} is empty. Skipping metrics.")
            continue
            
    
        test_preds_df['Y_hat_best'] = (test_preds_df['p_1'] >= best_thr).astype(int)
        test_preds_df.to_csv(os.path.join(full_exp_dir, f"test_preds_fold_{i}.csv"), index=False)

     
        labels, probs = test_preds_df['label'], test_preds_df['p_1']
    
        metrics_argmax = calculate_sota_metrics(labels, test_preds_df['Y_hat_argmax'], probs)
        
    
        metrics_best = calculate_sota_metrics(labels, test_preds_df['Y_hat_best'], probs)
        
        fold_metrics = {'fold': i, 'best_threshold_from_val': best_thr}
        for key, val in metrics_argmax.items():
       
            if key in ['auc', 'auprc']:
                fold_metrics[key] = val
            else:
                fold_metrics[f"{key}_argmax"] = val
        
        for key in ['f1_macro', 'accuracy', 'sensitivity', 'specificity']:
            fold_metrics[f"{key}_best"] = metrics_best[key] 
        
        all_fold_metrics.append(fold_metrics)
        
        print(f"  --> Test Results (Argmax/0.5 Thresh): AUC={fold_metrics.get('auc',-1):.4f}, F1={fold_metrics.get('f1_macro_argmax',-1):.4f}")
        print(f"  --> Test Results (Best Thresh):       AUC={fold_metrics.get('auc',-1):.4f}, F1={fold_metrics.get('f1_macro_best',-1):.4f}")

    print("\n======================= FINAL SUMMARY =======================")
    if all_fold_metrics:
        summary_df = pd.DataFrame(all_fold_metrics)
        summary_path = os.path.join(full_exp_dir, "summary_metrics.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved detailed summary to {summary_path}")
        
        key_cols = [c for c in summary_df.columns if c not in ['fold', 'best_threshold_from_val']]
        mean_std_summary = summary_df[key_cols].agg(['mean', 'std']).transpose()
        print("\nMean and STD of metrics across all folds:")
        print(mean_std_summary.round(4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SOTA Test script with dual metric reporting.')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--split_dir_name', type=str, required=True)
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--label_col', type=str, required=True)
    parser.add_argument('--positive_label', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--omic_input_dim', type=int, required=True)
    parser.add_argument('--wsi_encoder_dim', type=int, default=512)
    parser.add_argument('--rna_embedding_dim', type=int, default=512)
    parser.add_argument('--shared_embedding_dim', type=int, default=64)
    parser.add_argument('--num_memory', type=int, default=4)
    parser.add_argument('--cls_hidden_dim', type=int, default=128)
    args = parser.parse_args()
    args.split_dir = os.path.join('./splits', '5foldcv', args.split_dir_name)
    main(args)
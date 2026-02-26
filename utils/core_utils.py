from argparse import Namespace
from collections import OrderedDict
import os
import pickle 


import numpy as np


import torch


from datasets.dataset_generic import save_splits

from utils.utils import *






class EarlyStopping:
    # This class is well-defined, but we'll add a small improvement
    # to save the best model's score.
    def __init__(self, patience=10, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
            stop_epoch (int): Earliest epoch possible for stopping.
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, epoch, val_metric, model, ckpt_name='checkpoint.pt'):
        # We now assume val_metric is a score where higher is better (like AUC)
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_metric, model, ckpt_name):
        if self.verbose:
            print(f'Validation metric improved ({self.best_score:.4f}). Saving model ...')
        torch.save(model.state_dict(), ckpt_name)




from collections import defaultdict

from models.mkd import MKD
from utils.Mkd_train_utils import mkd_train_loop, mkd_validate_loop,PrototypicalAlignmentLoss



def train(datasets: tuple, cur: int, args: Namespace):

    if args.model_name == 'mkd':
        print("\n>>> Running MKD Training <<<")
        return train_mkd(datasets, cur, args)
    
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")

from tqdm import tqdm


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [
        N / len(dataset.slide_cls_ids[c]) if len(dataset.slide_cls_ids[c]) > 0 else 1e-8
        for c in range(len(dataset.slide_cls_ids))
    ]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]
    return torch.DoubleTensor(weight)
import os, random, numpy as np, torch


@torch.no_grad()
def bootstrap_codebooks(dataloader, model, device, max_per_class=10000, kmeans_iters=20):
    model.eval()
    pos_feats, neg_feats = [], []
    for (wsi_data_tuple, _, label, _) in dataloader:
        wsi_x, edge_index = wsi_data_tuple
        wsi_x, edge_index = wsi_x.to(device), edge_index.to(device)
        contextual = model.wsi_encoder((wsi_x, edge_index))                     # (P, Dws)
        patch_proj = F.normalize(model.wsi_patch_projector(contextual), p=2, dim=-1).detach().cpu()
        if int(label.item()) == 1:
            pos_feats.append(patch_proj)
        else:
            neg_feats.append(patch_proj)
        if (sum(x.size(0) for x in pos_feats) >= max_per_class and
            sum(x.size(0) for x in neg_feats) >= max_per_class):
            break

    import torch
    def kmeans(feats, K, iters=20):
        x = torch.cat(feats, dim=0)       # (N,S)
        x = F.normalize(x, dim=-1)
        N = x.size(0)
        idx = torch.randperm(N)[:K]
        c = x[idx].clone()                # (K,S)
        for _ in range(iters):
            sim = x @ c.T               
            a = sim.argmax(dim=1)
            for k in range(K):
                m = (a == k)
                if m.any():
                    c[k] = F.normalize(x[m].mean(0, keepdim=True), dim=-1)
        return c

    K = model.K
    if len(pos_feats):
        c_pos = kmeans(pos_feats, K, iters=kmeans_iters)
        model.mem_pos.codebook.copy_(c_pos.to(model.mem_pos.codebook.device))
    if len(neg_feats):
        c_neg = kmeans(neg_feats, K, iters=kmeans_iters)
        model.mem_neg.codebook.copy_(c_neg.to(model.mem_neg.codebook.device))

  
    model.mem_pos.codebook.copy_(F.normalize(model.mem_pos.codebook, dim=-1))
    model.mem_neg.codebook.copy_(F.normalize(model.mem_neg.codebook, dim=-1))


def train_mkd(datasets: tuple, cur: int, args: Namespace):

    print(f"\n--- Training MKD for Fold {cur} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_split, val_split = datasets
    print(f"Training on {len(train_split)} samples, Validating on {len(val_split)} samples.")
    save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    # --- START DEBUGGING BLOCK ---
    print("\n\n--- [DEBUG] Verifying received dataset objects in train() ---")
    print(f"  train_split object type: {type(train_split)}")
    print(f"  train_split length: {len(train_split)}")
    if hasattr(train_split, 'slide_data'):
        print("  Label distribution in TRAIN split:")
        print(train_split.slide_data['label'].value_counts())
    
    print(f"  val_split object type: {type(val_split)}")
    print(f"  val_split length: {len(val_split)}")
    if hasattr(val_split, 'slide_data'):
        print("  Label distribution in VAL split:")
        print(val_split.slide_data['label'].value_counts())
    print("----------------------------------------------------------\n")
    # DataLoaders - batch_size=1 for MIL


    train_weights = make_weights_for_balanced_classes_split(train_split)  
    train_sampler = WeightedRandomSampler(
        weights=train_weights, 
        num_samples=len(train_split),   
        replacement=True,              

    )

    train_loader = DataLoader(
    train_split,
    batch_size=1,
    sampler=train_sampler,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
  
  
)
    
    val_loader = DataLoader(val_split, batch_size=1, shuffle=False, num_workers=4)



    print("Initializing MKD model...")
    model = MKD(
        rna_input_dim=args.omic_input_dim,
        wsi_input_dim=1536,  # Standard feature dim
        wsi_encoder_dim=args.wsi_encoder_dim,
        rna_embedding_dim=args.rna_embedding_dim,
        shared_embedding_dim=args.shared_embedding_dim,
        num_memory_per_class=args.num_memory,
        cls_hidden_dim=args.cls_hidden_dim
    ).to(device)
    print(model)
    bootstrap_codebooks(train_loader, model, device, max_per_class=50000, kmeans_iters=20)
    print("start pos/neg codebook kmeans")
    optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-4
        )




    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)
    early_stopper = EarlyStopping(patience=args.patience, stop_epoch=args.stop_epoch, verbose=True)
    log_dir = os.path.join(args.results_dir, "runs", f"s_{cur}_MKD") 







    # Training Loop
    for epoch in range(args.max_epochs):

        train_hparams = {
            "alpha": 0.3, 
            "task_weight": 0.5,
            "alpha_rna": 0.05,       
            "alpha_wsi": 0.2, 
            "margin": 0.3, 
            "gc": args.gc,
            "recon_weight": 0.05,
            "results_dir":args.results_dir,
            "diag_csv": os.path.join(args.results_dir, f"s_{cur}_MKD_diag.csv"), 

        }
        
        
        train_results = mkd_train_loop(epoch, model, train_loader, optimizer, train_hparams, device)
        
        print(
            f"MKD | Epoch {epoch}: Train ["
            f"Task_L: {train_results['loss_task']:.4f}, "
            f"Align_rna: {train_results['loss_align_rna']:.4f}, "
            f"Align_wsi: {train_results['loss_align_wsi']:.4f}, "
            f"Mem_loss: {train_results['mem_loss']:.4f}, "
            f"Recon_L: {train_results['loss_recon']:.4f}, "  
            f"AUC: {train_results['auc']:.4f}]"
        )
 
        # Validation
        val_results = mkd_validate_loop(model, val_loader, device)
        val_auc = val_results["auc"]
        val_loss = val_results["loss_task"]
        
        print(f"Validation | Epoch {epoch}: Val Loss (Task): {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        scheduler.step(val_auc)
        

        ckpt_path = os.path.join(args.results_dir, f"s_{cur}_mkd_checkpoint.pt")
        early_stopper(epoch, val_auc, model, ckpt_name=ckpt_path)
   
        if early_stopper.early_stop:
            print("MKD training early stopped.")
            break


    print(f"Finished training MKD for Fold {cur}. Best Val AUC: {early_stopper.best_score:.4f}")
    
    return early_stopper.best_score





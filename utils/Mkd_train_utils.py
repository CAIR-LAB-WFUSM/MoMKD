
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import csv 
import os



class PrototypicalAlignmentLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        """
        margin: margin value used in the triplet-like margin loss
        alpha_rna: weight for RNA -> prototypes component
        alpha_wsi: weight for WSI -> prototypes component
        """
        super().__init__()
        self.margin = margin
    def _agg(self, sim: torch.Tensor) -> torch.Tensor:

        tau=5
        return (1.0 / tau) * torch.logsumexp(tau * sim, dim=1)
    
    def forward(self, rna_proj: torch.Tensor, wsi_proj: torch.Tensor,pos_centers: torch.Tensor, neg_centers: torch.Tensor, label: torch.Tensor) -> torch.Tensor:

        beta=20
        # Similarities: (1, K) each
        s_rna_pos = self._agg((rna_proj @ pos_centers.T))  # (1,)
        s_rna_neg = self._agg( (rna_proj @ neg_centers.T))
        s_wsi_pos = self._agg( (wsi_proj  @ pos_centers.T))
        s_wsi_neg = self._agg( (wsi_proj  @ neg_centers.T))
      
        y = int(label.item())
        if y == 1:
            
            loss_rna = F.softplus(beta * (self.margin - (s_rna_pos - s_rna_neg))) 
            loss_wsi = F.softplus(beta * (self.margin - (s_wsi_pos - s_wsi_neg))) 
        else:
            loss_rna = F.softplus(beta * (self.margin - (s_rna_neg - s_rna_pos))) 
            loss_wsi = F.softplus(beta * (self.margin - (s_wsi_neg - s_wsi_pos))) 
        return {
            'rna': loss_rna.mean(),
            'wsi': loss_wsi.mean()
        }

    








def mkd_train_loop(epoch, model, loader, optimizer, hparams, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    recon_loss_fn = nn.MSELoss()
    aligner = PrototypicalAlignmentLoss(
        margin=hparams.get('margin', 0.4),
       
    )
    task_w  = hparams.get('task_weight', 1.0)
    recon_weight = hparams.get('recon_weight', 0.5)
   
 
   
    
 
    total_task = total_mem =total_align_rna = total_align_wsi=total_recon= 0.0

    all_probs, all_labels = [], []
    
  
 
            
   

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for i, (wsi_data_tuple, rna_data, label, _) in enumerate(pbar):
        wsi_features, edge_index = wsi_data_tuple
        wsi_features, edge_index = wsi_features.to(device), edge_index.to(device)
        rna_data, label = rna_data.to(device), label.to(device)
        
        out = model.forward_train(rna_data, (wsi_features, edge_index), label)


        logits = out['logits']
        mem_loss = out['mem_loss']
        alpha_rna   = hparams.get('alpha_rna', 0.5)  
        alpha_wsi   = hparams.get('alpha_wsi', 2.0) 
        alpha_mem   = 0.1
        rna_proj, wsi_proj = out['rna_proj'], out['wsi_proj']
        posC, negC = out['pos_centers'], out['neg_centers']
        reconstructed_rna = out['reconstructed_rna']

        loss_task  = ce(logits, label)
        loss_recon = recon_loss_fn(reconstructed_rna, rna_data)
        aligner.margin = hparams.get('margin', aligner.margin)
        loss_align = aligner(rna_proj, wsi_proj, posC, negC, label)
        loss_align_rna_raw = loss_align['rna']
        loss_align_wsi_raw = loss_align['wsi']
  
      

        loss = task_w*loss_task  + alpha_mem* mem_loss +alpha_rna*loss_align_rna_raw+alpha_wsi*loss_align_wsi_raw+ recon_weight * loss_recon
  
        loss.backward()

        


        
        gc = hparams.get('gc', 0.0)
        if gc and gc > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gc)
        optimizer.step()

        optimizer.zero_grad()

        total_task += float(loss_task.item())
   
        total_align_rna += float(loss_align_rna_raw.item()) 
        total_align_wsi += float(loss_align_wsi_raw.item()) 
        total_mem   += float(mem_loss.item() if torch.is_tensor(mem_loss) else mem_loss)
        total_recon += float(loss_recon.item())

        probs = torch.softmax(logits.detach(), dim=-1)[:, 1].cpu().item()
        all_probs.append(probs)
        all_labels.append(label.cpu().item())

        pbar.set_postfix(
    L_task=loss_task.item(),           
    L_rna=loss_align_rna_raw.item(), 
    L_wsi=loss_align_wsi_raw.item(), 
    L_recon=loss_recon.item(),
    lr=optimizer.param_groups[0]['lr']
)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels))>1 else 0.5
    print(f"[Diag][Epoch {epoch}] AUC={auc:.4f}")

    
    return {
        "loss_task": total_task/len(loader),

        "mem_loss": total_mem/len(loader),
        "loss_align_rna": total_align_rna / len(loader),
        "loss_align_wsi": total_align_wsi / len(loader),"loss_recon": total_recon / len(loader),
        "auc": auc
    }



def mkd_validate_loop(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Validation loop: evaluate WSI-only inference (this mirrors deployment).
    Returns AUC and average task loss.
    """
    model.eval()
    criterion_task = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for wsi_data_tuple, _, label, _ in tqdm(loader, desc="[Validation]"):
            wsi_features, edge_index = wsi_data_tuple
            wsi_features, edge_index = wsi_features.to(device), edge_index.to(device)
            wsi_data_for_model = (wsi_features, edge_index)
            label = label.to(device)

            logits, _ = model.forward_inference(wsi_data_for_model)
            loss = criterion_task(logits, label)
            total_loss += loss.item()

            probs = torch.softmax(logits.detach(), dim=-1)[:, 1].cpu().item()
            all_probs.append(probs)
            all_labels.append(label.cpu().item())

    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5

    return {
        "auc": auc,
        "loss_task": total_loss / len(loader)
    }


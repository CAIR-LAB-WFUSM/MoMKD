import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from torch_geometric.nn import GATv2Conv
from vq.vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize


class RnaEncoder(nn.Module):
    """
    Lightweight MLP encoder for RNA.
    Input:  x shape (1, input_dim)
    Output: (1, embedding_dim)
    """
    def __init__(self, input_dim: int, embedding_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, input_dim)
        return self.encoder(x)  # (1, embedding_dim)


class RnaDecoder(nn.Module):
    """
    Symmetric decoder to reconstruct RNA from latent.
    Input:  z shape (1, embedding_dim)
    Output: (1, input_dim)
    """
    def __init__(self, output_dim: int, embedding_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)  # (1, output_dim)


class WsiEncoder(nn.Module):
    """
    Two-layer GATv2 encoder with residual connections and JK (max).
    Input:
        wsi_data_tuple = (patch_features, edge_index)
        patch_features: (P, Din) or (1, P, Din)
        edge_index:     (2, E) or (1, 2, E)
    Output:
        contextual_features: (P, H)
    """
    def __init__(self, input_dim: int = 1536, output_dim: int = 256, heads: int = 2, dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.dropout = dropout

        # GATv2 blocks
        self.gat1 = GATv2Conv(
            in_channels=input_dim,
            out_channels=output_dim // heads,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            edge_dim=None,
        )
        self.gat2 = GATv2Conv(
            in_channels=output_dim,
            out_channels=output_dim // heads,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            edge_dim=None,
        )

       
        self.res_proj1 = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.res_proj2 = nn.Identity()

        self.ln1 = nn.LayerNorm(output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.act = nn.ELU()

    def _prep_inputs(self, wsi_data_tuple: Tuple[torch.Tensor, torch.Tensor]):
        x, edge_index = wsi_data_tuple

        # patch_features: (1, P, D) -> (P, D)
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        elif x.dim() != 2:
            raise ValueError(f"Unsupported patch_features shape: {x.shape}")

        # edge_index: (1, 2, E) -> (2, E)
        if edge_index.dim() == 3 and edge_index.size(0) == 1:
            edge_index = edge_index.squeeze(0)
        elif edge_index.dim() != 2:
            raise ValueError(f"Unsupported edge_index shape: {edge_index.shape}")

        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        return x, edge_index

    def forward(self, wsi_data_tuple: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, edge_index = self._prep_inputs(wsi_data_tuple)  # x: (P, Din)

    
        h1 = self.gat1(x, edge_index)             # (P, H)
        h1 = self.ln1(h1)
        h1 = self.act(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h1 = h1 + self.res_proj1(x)               # residual

    
        h2 = self.gat2(h1, edge_index)            # (P, H)
        h2 = self.ln2(h2)
        h2 = self.act(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = h2 + self.res_proj2(h1)              # residual

      
        contextual = torch.max(h1, h2)            # (P, H)
        return contextual


class MKD(nn.Module):

    def __init__(
        self,
        rna_input_dim: int,
        wsi_input_dim: int = 1536,
        wsi_encoder_dim: int = 256,
        rna_embedding_dim: int = 256,
        shared_embedding_dim: int = 128,
        num_memory_per_class: int = 4,
        cls_hidden_dim: int = 128,
        attn_tau_init: float = 0.2,
    ):
        super().__init__()
        # --- Encoders ---
        self.rna_tower = RnaEncoder(input_dim=rna_input_dim, embedding_dim=rna_embedding_dim)
        self.wsi_encoder = WsiEncoder(input_dim=wsi_input_dim, output_dim=wsi_encoder_dim)
        self.rna_decoder = RnaDecoder(output_dim=rna_input_dim, embedding_dim=rna_embedding_dim)

        # --- Projectors ---
        self.rna_projector = nn.Linear(rna_embedding_dim, shared_embedding_dim)
        self.wsi_patch_projector = nn.Linear(wsi_encoder_dim, shared_embedding_dim)  # patch-level projection
        self.wsi_pool_projector = nn.Linear(wsi_encoder_dim, shared_embedding_dim)   # pooled WSI projection

   
        self.K = num_memory_per_class
        memory = dict(
            dim=shared_embedding_dim,
            codebook_size=self.K,
            accept_image_fmap=False,
            orthogonal_reg_weight=10,
            orthogonal_reg_max_codes=128,
            orthogonal_reg_active_codes_only=False,
            ema_update=False,
        )
        self.mem_pos = VectorQuantize(**memory)
        self.mem_neg = VectorQuantize(**memory)

     
        self.attn_log_tau = nn.Parameter(torch.log(torch.tensor(attn_tau_init)))

     
        self.wsi_classifier_proto = nn.Sequential(
            nn.LayerNorm(wsi_encoder_dim),
            nn.Linear(wsi_encoder_dim, cls_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(cls_hidden_dim, 2),
        )

    def _attn_temp(self) -> torch.Tensor:
      
        attn_tau = torch.exp(self.attn_log_tau).clamp(0.01, 2.0)
        return attn_tau

    def forward_train(self, rna_data: torch.Tensor, wsi_data_tuple: Tuple[torch.Tensor, torch.Tensor], label: torch.Tensor):

        # 1) Encode
        rna_emb = self.rna_tower(rna_data)  # (1, Drna)
        contextual = self.wsi_encoder(wsi_data_tuple)  # (P, Dws)

        rna_proj = F.normalize(self.rna_projector(rna_emb), p=2, dim=-1)              # (1, S)
        patch_proj = F.normalize(self.wsi_patch_projector(contextual), p=2, dim=-1)   # (P, S)
        reconstructed_rna = self.rna_decoder(rna_emb)                                 # (1, Drna)

        # 2) Label-conditional VQ update
        is_pos = bool(label.view(-1)[0].item() == 1)

        if is_pos:
          
            _, _, mem_loss = self.mem_pos(patch_proj.unsqueeze(0), freeze_codebook=False)
            with torch.no_grad():
             
                _, _, _ = self.mem_neg(patch_proj.unsqueeze(0), freeze_codebook=True)
        else:
           
            _, _, mem_loss = self.mem_neg(patch_proj.unsqueeze(0), freeze_codebook=False)
            with torch.no_grad():
              
                _, _, _ = self.mem_pos(patch_proj.unsqueeze(0), freeze_codebook=True)
        attn_tau = self._attn_temp()
        posC = F.normalize(self.mem_pos.codebook, p=2, dim=-1)  # (K, S)
        negC = F.normalize(self.mem_neg.codebook, p=2, dim=-1)  # (K, S)

        posC_attn = posC.detach()
        negC_attn = negC.detach()

        sim_pos = patch_proj @ posC_attn.T  # (P, K)
        sim_neg = patch_proj @ negC_attn.T  # (P, K)

    
        idx_pos = sim_pos.argmax(dim=1)  # (P,)
        idx_neg = sim_neg.argmax(dim=1)  # (P,)

        s_pos = sim_pos.max(dim=1).values  # (P,)
        s_neg = sim_neg.max(dim=1).values  # (P,)

        patch_scores = (s_pos - s_neg).unsqueeze(1)                    # (P, 1)
        attention = F.softmax(patch_scores / (attn_tau + 1e-9), dim=0) # (P, 1)

      
        V_wsi = torch.sum(attention * contextual, dim=0, keepdim=True) # (1, Dws)
        logits = self.wsi_classifier_proto(V_wsi)                      # (1, 2)

   
        wsi_proj = F.normalize(self.wsi_pool_projector(V_wsi), p=2, dim=-1)  # (1, S)

        return {
            "logits": logits,
            "rna_proj": rna_proj,
            "wsi_proj": wsi_proj,
            "attention": attention.detach(),
            "pos_centers": posC,
            "neg_centers": negC,
            "mem_loss":mem_loss,
            "reconstructed_rna": reconstructed_rna,
            "idx_pos": idx_pos,              
            "idx_neg": idx_neg, 

        }

    @torch.no_grad()
    def forward_inference(self, wsi_data_tuple: Tuple[torch.Tensor, torch.Tensor]):

        contextual = self.wsi_encoder(wsi_data_tuple)                     # (P, Dws)
        patch_proj = F.normalize(self.wsi_patch_projector(contextual), p=2, dim=-1)

        attn_tau = self._attn_temp()
        posC = F.normalize(self.mem_pos.codebook, p=2, dim=-1)
        negC = F.normalize(self.mem_neg.codebook, p=2, dim=-1)

        sim_pos = patch_proj @ posC.T
        sim_neg = patch_proj @ negC.T

        s_pos = sim_pos.max(dim=1).values
        s_neg = sim_neg.max(dim=1).values

        patch_scores = (s_pos - s_neg).unsqueeze(1)
        attention = F.softmax(patch_scores / (attn_tau + 1e-9), dim=0)

        V_wsi = torch.sum(attention * contextual, dim=0, keepdim=True)
        logits = self.wsi_classifier_proto(V_wsi)

        return logits, attention

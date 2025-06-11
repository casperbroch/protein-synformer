import torch 
from torch import nn
from synformer.data.common import ProjectionBatch

from .base import BaseEncoder, EncoderOutput


class ProteinIntermediateEncoder(BaseEncoder):
    def __init__(self, 
                 d_model: int = 512,  # aka d_latent 
                 nhead: int = 8,
                 dim_feedforward: int = 2048,  # usually 4 * d_model 
                 d_protein: int = 1152, 
                 num_layers: int = 1,
                 num_latents: int = 32,
                 output_norm: bool = False):
        super().__init__()
        self._dim = d_model 
        # Our protein embeddings have 1152 dimensions, but the decoder (trained weights) expects 768 dimensions,
        # so we use a linear layer to project them down to 768 dimensions.
        self.proj = nn.Linear(d_protein, d_model) 
        # Learnable latent array (latent_seq_len x latent_dim)
        self.latents = nn.Parameter(
            torch.randn(num_latents, d_model)
        )
        self.enc = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, 
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=(
                nn.LayerNorm(d_model) 
                if output_norm 
                else None
            )
        )

    @property
    def dim(self) -> int:
        return self._dim
    
    def forward(self, batch: ProjectionBatch):
        if "protein_embeddings" not in batch:
            raise ValueError("protein_embeddings must be in batch")
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
        protein_embeddings = batch["protein_embeddings"]  # (batch_size, protein_seq_len, d_protein)
        bsz = protein_embeddings.size(0)  # batch size 

        projected_embeddings = self.proj(protein_embeddings)

        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)  # add batch dimension 
        code = self.enc(tgt=latents, memory=projected_embeddings)  

        # code_padding_mask = torch.zeros([bsz, 0], dtype=torch.bool, device=device)
        code_padding_mask = None

        return EncoderOutput(code, code_padding_mask)

import torch 
from torch import nn
from synformer.data.common import ProjectionBatch

from .base import BaseEncoder, EncoderOutput


class ProteinEncoder(BaseEncoder):
    def __init__(self, 
                 d_model: int = 512,
                 d_protein: int = 1152):
        super().__init__()
        self._dim = d_model 
        # Project protein embeddings from d_protein (e.g. 1152) to d_model (e.g. 512 or 768)
        self.enc = nn.Linear(d_protein, d_model) 

    @property
    def dim(self) -> int:
        return self._dim
    
    def forward(self, batch: ProjectionBatch):
        # Ensure protein embeddings are in the batch
        if "protein_embeddings" not in batch:
            raise ValueError("protein_embeddings must be in batch")
        
        # Get device from any tensor in the batch
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break

        # Shape: (batch_size, seq_len, d_protein)
        protein_embeddings = batch["protein_embeddings"]

        # Project to (batch_size, seq_len, d_model)
        code = self.enc(protein_embeddings)

        # No padding mask used here
        code_padding_mask = None

        # Return encoded output
        return EncoderOutput(code, code_padding_mask)

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
        # !!
        # Our protein embeddings have 1152 dimensions, but the decoder (trained weights) expects 768 dimensions,
        # so we use a linear layer to project them down to 768 dimensions.
        # TODO: experiment whether more complex projection is better
        self.enc = nn.Linear(d_protein, d_model) 
        # !!

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
        protein_embeddings = batch["protein_embeddings"]
        # !!
        code = self.enc(protein_embeddings)
        # !!
        # bsz = code.size(0)  # batch size 
        # code_padding_mask = torch.zeros([bsz, 0], dtype=torch.bool, device=device)
        code_padding_mask = None
        return EncoderOutput(code, code_padding_mask)

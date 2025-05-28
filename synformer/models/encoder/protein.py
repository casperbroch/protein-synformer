import torch 
from synformer.data.common import ProjectionBatch

from .base import BaseEncoder, EncoderOutput


class ProteinEncoder(BaseEncoder):
    def __init__(self, d_model: int):
        super().__init__()
        self._dim = d_model 

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
        code = batch["protein_embeddings"]
        bsz = code.size(0)  # batch size 
        code_padding_mask = torch.zeros([bsz, 0], dtype=torch.bool, device=device)
        return EncoderOutput(code, code_padding_mask)

from .base import BaseEncoder, NoEncoder
from .graph import GraphEncoder
from .smiles import SMILESEncoder
from .protein import ProteinEncoder


def get_encoder(t: str, cfg) -> BaseEncoder:
    if t == "smiles":
        return SMILESEncoder(**cfg)
    elif t == "graph":
        return GraphEncoder(**cfg)
    elif t == "protein":
        return ProteinEncoder(**cfg)
    elif t == "none":
        return NoEncoder(**cfg)
    else:
        raise ValueError(f"Unknown encoder type: {t}")

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import nn
from lora_pytorch import LoRA

from synformer.data.common import TokenType
from synformer.models.transformer.positional_encoding import PositionalEncoding


def _SimpleMLP(dim_in: int, dim_out: int, dim_hidden: int) -> Callable[[torch.Tensor], torch.Tensor]:
    return nn.Sequential(
        nn.Linear(dim_in, dim_hidden),
        nn.ReLU(),
        nn.Linear(dim_hidden, dim_hidden),
        nn.ReLU(),
        nn.Linear(dim_hidden, dim_out),
    )


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 6,
        pe_max_len: int = 32,
        output_norm: bool = False,
        fingerprint_dim: int = 2048,
        num_reaction_classes: int = 120,
        decoder_only: bool = False,
        lora: bool = False,
        lora_rank: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.in_token = nn.Embedding(max(TokenType) + 1, d_model)
        self.in_reaction = nn.Embedding(num_reaction_classes, d_model)
        self.in_fingerprint = _SimpleMLP(fingerprint_dim, d_model, dim_hidden=d_model * 2)
        self.pe_dec = PositionalEncoding(d_model=d_model, max_len=pe_max_len)
        self.decoder_only = decoder_only
        if decoder_only:
            self.dec = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
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
                ),
            )
        else:
            # !!!
            self.dec = nn.TransformerDecoder(
                decoder_layer=nn.TransformerDecoderLayer(
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
                ),
            )
            # !!!
        # !!!
        if lora:
            # This wraps the decoder in a LoRA module, allowing for low-rank adaptation.
            # It modifies self.dec, so it now contains the LoRA layers.
            self.lora_dec = LoRA.from_module(
                self.dec, 
                rank=lora_rank
            )
        # !!!

    def get_empty_code(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        code = torch.zeros([batch_size, 0, self.model_dim], dtype=dtype, device=device)
        code_padding_mask = torch.zeros([batch_size, 0], dtype=torch.bool, device=device)
        return code, code_padding_mask

    def embed(
        self,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
    ) -> torch.Tensor:
        emb_token = self.in_token(token_types)
        emb_rxn = self.in_reaction(rxn_indices)
        emb_fingerprint = self.in_fingerprint(reactant_fps)
        token_types_expand = token_types.unsqueeze(-1).expand([token_types.size(0), token_types.size(1), self.d_model])
        emb_token = torch.where(token_types_expand == TokenType.REACTION, emb_rxn, emb_token)
        emb_token = torch.where(token_types_expand == TokenType.REACTANT, emb_fingerprint, emb_token)
        emb_token = self.pe_dec(emb_token)
        return emb_token

    def forward(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
        token_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, seqlen = token_types.size()
        # print("bsz", bsz)  # e.g. 10
        # print("seqlen", seqlen)  # e.g. 24
        # print("token_types", token_types.shape)  # [bsz, n_tokens] (n_tokens aka seqlen)
        # print("rxn_indices", rxn_indices.shape)  # [bsz, n_tokens]
        # print("reactant_fps", reactant_fps.shape)  # [bsz, n_tokens, n_morgan_bits]
        # print("code", code.shape if code is not None else None)  # [bsz, seqlen, d_model] or None
        # print("code_padding_mask", code_padding_mask.shape if code_padding_mask is not None else None)  # [bsz, seqlen, d_model] or None
        x = self.embed(token_types, rxn_indices, reactant_fps)  # [bsz, n_tokens, d_model]
        # print("x", x.shape)  # e.g. [10, 24, 1] 
        # print("token_padding_mask", token_padding_mask.shape if token_padding_mask is not None else None)  # [bsz, n_tokens] or None
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=x.size(1),
            dtype=x.dtype,
            device=x.device,
        )
        tgt_key_padding_mask = (
            torch.zeros(
                [bsz, seqlen],
                dtype=causal_mask.dtype,
                device=causal_mask.device,
            ).masked_fill_(token_padding_mask, -torch.finfo(causal_mask.dtype).max)
            if token_padding_mask is not None
            else None
        )
        # print("tgt_key_padding_mask", tgt_key_padding_mask.shape)  # [bsz, n_tokens]
        if self.decoder_only:
            y: torch.Tensor = self.dec(
                src=x,
                src_key_padding_mask=tgt_key_padding_mask,
                mask=causal_mask,
            )
        else:
            if code is None:
                code, code_padding_mask = self.get_empty_code(bsz, device=reactant_fps.device, dtype=reactant_fps.dtype)
            # !!!
            y = self.dec(
                tgt=x,
                memory=code,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=code_padding_mask,
            )  # (bsz, seq_len, d_model)
            # !!!
        return y

    if TYPE_CHECKING:

        def __call__(
            self,
            code: torch.Tensor | None,
            code_padding_mask: torch.Tensor | None,
            token_types: torch.Tensor,
            rxn_indices: torch.Tensor,
            reactant_fps: torch.Tensor,
            token_padding_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            ...

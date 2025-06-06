import os
import pickle
import pandas as pd 
import numpy as np
# import random
from typing import cast

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
# from synformer.chem.stack import create_stack_step_by_step
from synformer.chem.mol import FingerprintOption, Molecule 
from synformer.utils.train import worker_init_fn
from synformer.data.common import TokenType

from .collate import (
    apply_collate,
    collate_1d_features,
    # collate_2d_tokens,
    collate_padding_masks,
    collate_tokens,
)
from .common import ProjectionBatch, ProjectionData  #, create_data


class Collater:
    """
    Batch assembly:
    combines multiple examples into a single batch, including doing things like padding 
    """
    def __init__(self, 
                 max_protein_len: int = 2010,  # proteins are padded to max_protein_len
                 max_num_tokens: int = 24):
        super().__init__()
        self.max_protein_len = max_protein_len
        self.max_num_tokens = max_num_tokens
        self.spec_protein = {
            "protein_embeddings": collate_1d_features,  # not sure if collate_1d_features is the correct one
            # "protein_padding_mask": collate_padding_masks,  # currently not used
        }
        self.spec_tokens = {
            "token_types": collate_tokens,
            "rxn_indices": collate_tokens,
            "reactant_fps": collate_1d_features,
            "token_padding_mask": collate_padding_masks,
        }

    def __call__(self, data_list: list[ProjectionData]) -> ProjectionBatch:
        data_list_t = cast(list[dict[str, torch.Tensor]], data_list)
        batch = {
            **apply_collate(self.spec_protein, data_list_t, max_size=self.max_protein_len),
            **apply_collate(self.spec_tokens, data_list_t, max_size=self.max_num_tokens),
            "mol_seq": [d["mol_seq"] for d in data_list],
            "rxn_seq": [d["rxn_seq"] for d in data_list],
        }
        return cast(ProjectionBatch, batch)
        

class ProjectionDataset(Dataset[ProjectionData]):
    def __init__(
        self,
        rxn_matrix: ReactantReactionMatrix,  
        fpindex: FingerprintIndex, 
        protein_molecule_pairs: np.ndarray,  
        protein_embeddings: dict, 
        synthetic_pathways: dict,  
        max_num_reactions: int = 5,
        init_stack_weighted_ratio: float = 0.0,
        fp_option: FingerprintOption = FingerprintOption()
    ) -> None:
        super().__init__()
        self._max_num_reactions = max_num_reactions
        self._fpindex = fpindex
        self._fp_option = fp_option 
        self._rxn_matrix = rxn_matrix
        self._protein_molecule_pairs = protein_molecule_pairs 
        self._protein_embeddings = protein_embeddings 
        self._synthetic_pathways = synthetic_pathways 
        self._init_stack_weighted_ratio = init_stack_weighted_ratio

    def __len__(self) -> int:
        return len(self._protein_molecule_pairs)

    def __getitem__(self, idx):
        """
        Iterate over protein-molecule pairs and yield appropriate data for each pair,
        which includes both the encoder data (protein embeddings) and the decoder data 
        (token types, reaction tokens, reactant fingerprints).

        In the original code, it did something like this:
        while True:
            for stack in create_stack_step_by_step(...):
                ...
                data = create_data(...)
                yield data

        Here some examples and shapes from original data (SynFormer-ED, generating synthetic pathways
        using templates) for reference:

        # Example:
        mol_seq_full (<synformer.chem.mol.Molecule object at 0x17e0a6e90>, 
                      <synformer.chem.mol.Molecule object at 0x17cd06fb0>, 
                      <synformer.chem.mol.Molecule object at 0x30cbe4cd0>)  # notice: None when it's a reaction 
        mol_idx_seq_full [128510, 43161, None]  
        rxn_seq_full (None, None, <synformer.chem.reaction.Reaction object at 0x17fa29510>)
        rxn_idx_seq_full [None, None, 59]  # notice: None when it's a building block
        
        # Types:
        mol_seq_full <class 'tuple'>
        mol_idx_seq_full <class 'list'>
        rxn_seq_full <class 'tuple'>
        rxn_idx_seq_full <class 'list'>
        product <class 'synformer.chem.mol.Molecule'>
        
        # data = create_data(...): 
        {
            # Molecule:
            'atoms': tensor([...]),  # int [n_atoms]
            'bonds': tensor([...]),  # int [n_atoms, n_atoms]
            'smiles': tensor([...]),  # int [n_smiles]  # generally a bit more than n_atoms 
            'atom_padding_mask': tensor([...]),  # bool [n_atoms]  
            
            # Synthetic pathway:
            'token_types': tensor([1, 3, 3, 2, 0]),  # example [n_tokens]  # number of pathway tokens
            'rxn_indices': tensor([ 0,  0,  0, 59,  0]),  # example; notice: 0 when it's a building block [n_tokens]
            'reactant_fps': tensor([...]),  # !! zero-vector when it's a reaction [n_tokens, fp_dims]  # fp_dims, e.g. 256 or 2048 
            'token_padding_mask': tensor([...]),  # bool [n_tokens]
            
            # Auxiliary:
            'mol_seq': (<synformer.chem.mol.Molecule object>, 
                        <synformer.chem.mol.Molecule object>, 
                        <synformer.chem.mol.Molecule object>),  # ?
            'rxn_seq': (None, None, <synformer.chem.reaction.Reaction object>)  # ?
        } 
        """
        smiles, protein_id = self._protein_molecule_pairs[idx]
        if smiles in self._synthetic_pathways and protein_id in self._protein_embeddings:
            pathway = torch.tensor(self._synthetic_pathways[smiles], dtype=torch.long)
            token_types = pathway[:, 0]
            rxn_indices = torch.where(pathway[:,0] == TokenType.REACTION, pathway[:,1], 0)  # extract rxn_indices; fill others with 0 
            reactant_indices = torch.where(pathway[:,0] == TokenType.REACTANT, pathway[:,1], 0)  # extract reactant_indices; fill others with 0 
            # If reactant_idx == 0, it's a reaction, so we use a zero-vector for its fingerprint
            reactant_fps = torch.tensor(
                np.array([
                    self._fpindex._fp[reactant_idx] 
                    if reactant_idx != 0 
                    else np.zeros(self._fp_option.morgan_n_bits) 
                    for reactant_idx in reactant_indices 
                ]),
                dtype=torch.float32
            )
            protein_embeddings = self._protein_embeddings[protein_id].to(torch.float32)
            token_padding_mask = torch.zeros_like(token_types, dtype=torch.bool)
            # I assume it's n_tokens-2 elements (minus start, end tokens); see example below: 3 elements 
            #  so I skip the start and end tokens
            mol_seq_full = [
                self._fpindex.molecules[reactant_idx]  
                if reactant_idx != 0 
                else None
                for reactant_idx in reactant_indices[1:-1] 
            ]
            rxn_seq_full = [
                self._rxn_matrix.reactions[rxn_idx]  # rxn_matrix.reactions[rxn_idx] returns Reaction object
                if rxn_idx != 0 
                else None
                for rxn_idx in rxn_indices[1:-1] 
            ]
            data: "ProjectionData" = {
                # Encoder data (protein):
                "protein_embeddings": protein_embeddings,
                
                # Decoder data (synthetic pathway):
                "token_types": token_types,
                "rxn_indices": rxn_indices,
                "reactant_fps": reactant_fps,
                "token_padding_mask": token_padding_mask,
                
                # Auxiliary data:
                "mol_seq": mol_seq_full, 
                "rxn_seq": rxn_seq_full
            }
            return data 


class ProjectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        batch_size: int,
        num_workers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_options = kwargs

    def setup(self, stage: str | None = None) -> None:
        trainer = self.trainer
        if trainer is None:
            raise RuntimeError("The trainer is missing.")

        if not os.path.exists(self.config.chem.rxn_matrix):
            raise FileNotFoundError(
                f"Reaction matrix not found: {self.config.chem.rxn_matrix}. "
                "Please generate the reaction matrix before training."
            )
        if not os.path.exists(self.config.chem.fpindex):
            raise FileNotFoundError(
                f"Fingerprint index not found: {self.config.chem.fpindex}. "
                "Please generate the fingerprint index before training."
            )
        if not os.path.exists(self.config.chem.protein_molecule_pairs_train_path):
            raise FileNotFoundError(
                f"Protein-molecule pairs (train) not found: {self.config.chem.protein_molecule_pairs_train_path}."
            )
        if not os.path.exists(self.config.chem.protein_molecule_pairs_val_path):
            raise FileNotFoundError(
                f"Protein-molecule pairs (val) not found: {self.config.chem.protein_molecule_pairs_val_path}."
            )
        if not os.path.exists(self.config.chem.protein_embedding_path):
            raise FileNotFoundError(
                f"Protein embeddings not found: {self.config.chem.protein_embedding_path}."
            )
        if not os.path.exists(self.config.chem.synthetic_pathways_path):
            raise FileNotFoundError(
                f"Synthetic pathways not found: {self.config.chem.synthetic_pathways_path}."
            )

        with open(self.config.chem.rxn_matrix, "rb") as f:
            rxn_matrix = pickle.load(f)

        with open(self.config.chem.fpindex, "rb") as f:
            fpindex = pickle.load(f)

        with open(self.config.chem.protein_molecule_pairs_train_path, "rb") as f:
            protein_molecule_pairs_train = pd.read_csv(f).to_numpy()
            print(len(protein_molecule_pairs_train), "\t", "protein-molecule pairs (train)")

        with open(self.config.chem.protein_molecule_pairs_val_path, "rb") as f:
            protein_molecule_pairs_val = pd.read_csv(f).to_numpy()
            print(len(protein_molecule_pairs_val), "\t", "protein-molecule pairs (val)")

        # Always map_location="cpu" so that these embeddings live on CPU
        with open(self.config.chem.protein_embedding_path, "rb") as f:
            protein_embeddings = torch.load(f, map_location=torch.device("cpu"))
            print(len(protein_embeddings), "\t", "protein embeddings (loaded on CPU)")

        with open(self.config.chem.synthetic_pathways_path, "rb") as f:
            synthetic_pathways = torch.load(f, map_location=torch.device("cpu"))
            print(len(synthetic_pathways), "\t", "synthetic pathways")

        self.train_dataset = ProjectionDataset(
            rxn_matrix=rxn_matrix,
            fpindex=fpindex,
            protein_molecule_pairs=protein_molecule_pairs_train,
            protein_embeddings=protein_embeddings,
            synthetic_pathways=synthetic_pathways,
            fp_option=self.config.chem.fp_option,
            **self.dataset_options,
        )

        self.val_dataset = ProjectionDataset(
            rxn_matrix=rxn_matrix,
            fpindex=fpindex,
            protein_molecule_pairs=protein_molecule_pairs_val,
            protein_embeddings=protein_embeddings,
            synthetic_pathways=synthetic_pathways,
            fp_option=self.config.chem.fp_option,
            **self.dataset_options,
        )

    def train_dataloader(self):
        use_workers = self.num_workers > 0
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=Collater(),
            worker_init_fn=worker_init_fn,
            persistent_workers=use_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=Collater(),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            pin_memory=True,
        )


# TODO: modify to enable protein embeddings and synthetic pathways
"""
class ProjectionDataModuleForSample(pl.LightningDataModule):
    def __init__(
        self,
        config,
        batch_size: int,
        num_workers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_options = kwargs

    def setup(self, stage: str | None = None) -> None:
        if not os.path.exists(self.config.chem.rxn_matrix):
            raise FileNotFoundError(
                f"Reaction matrix not found: {self.config.chem.rxn_matrix}. "
                "Please generate the reaction matrix before training."
            )
        if not os.path.exists(self.config.chem.fpindex):
            raise FileNotFoundError(
                f"Fingerprint index not found: {self.config.chem.fpindex}. "
                "Please generate the fingerprint index before training."
            )
        if not os.path.exists(self.config.chem.protein_embedding_path):
            raise FileNotFoundError(
                f"Protein embeddings not found: {self.config.chem.protein_embedding_path}. "
            )

        with open(self.config.chem.rxn_matrix, "rb") as f:
            rxn_matrix = pickle.load(f)

        with open(self.config.chem.fpindex, "rb") as f:
            fpindex = pickle.load(f)
        
        with open(self.config.chem.protein_embedding_path, "rb") as f:
            protein_embeddings = torch.load(f)

        self.train_dataset = ProjectionDataset(
            rxn_matrix=rxn_matrix,
            fpindex=fpindex,
            protein_embeddings=protein_embeddings,
            # virtual_length=self.config.train.val_check_interval * self.batch_size,  # len(protein_molecule_pairs_train) ?
            **self.dataset_options,
        )
        self.val_dataset = ProjectionDataset(
            rxn_matrix=rxn_matrix,
            fpindex=fpindex,
            protein_embeddings=protein_embeddings,
            # virtual_length=self.batch_size,
            **self.dataset_options,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=Collater(),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=Collater(),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            pin_memory=True,
        )
""" 

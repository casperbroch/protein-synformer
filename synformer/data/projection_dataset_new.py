import os
import pickle
import pandas as pd 
import numpy as np
import random
from typing import cast

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
# from synformer.chem.stack import create_stack_step_by_step
from synformer.chem.mol import FingerprintOption, Molecule
from synformer.utils.train import worker_init_fn

from .collate import (
    apply_collate,
    collate_1d_features,
    collate_2d_tokens,
    collate_padding_masks,
    collate_tokens,
)
from .common import ProjectionBatch, ProjectionData  #, create_data


class Collater:
    def __init__(self, 
                 max_protein_len: int = 10000,  # not sure what value makes sense here, if any
                 # max_num_atoms: int = 96, 
                 # max_smiles_len: int = 192, 
                 max_num_tokens: int = 24):
        super().__init__()
        self.max_protein_len = max_protein_len
        # self.max_num_atoms = max_num_atoms
        # self.max_smiles_len = max_smiles_len
        self.max_num_tokens = max_num_tokens
        self.spec_protein = {
            "protein_embeddings": collate_1d_features,  # not sure if collate_1d_features is the correct one
            # "protein_padding_mask": collate_padding_masks,  # currently not used
        }
        """
        self.spec_atoms = {
            "atoms": collate_tokens,
            "bonds": collate_2d_tokens,
            "atom_padding_mask": collate_padding_masks,
        }
        self.spec_smiles = {"smiles": collate_tokens}
        """
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
            # **apply_collate(self.spec_atoms, data_list_t, max_size=self.max_num_atoms),
            # **apply_collate(self.spec_smiles, data_list_t, max_size=self.max_smiles_len),
            **apply_collate(self.spec_tokens, data_list_t, max_size=self.max_num_tokens),
            # "mol_seq": [d["mol_seq"] for d in data_list],
            # "rxn_seq": [d["rxn_seq"] for d in data_list],
        }
        return cast(ProjectionBatch, batch)
        

class ProjectionDataset(IterableDataset[ProjectionData]):
    def __init__(
        self,
        # reaction_matrix: ReactantReactionMatrix,  # rxn_matrix 
        fpindex: FingerprintIndex,  # fpindex
        protein_molecule_pairs: np.ndarray,  
        protein_embeddings: dict, 
        synthetic_pathways: dict,  
        max_num_atoms: int = 80,  # still needed for stack?? even if we don't pass atoms/bonds/smiles to encoder? 
        # max_smiles_len: int = 192,
        max_num_reactions: int = 5,
        virtual_length: int = 65536,
        # config.data: 
        init_stack_weighted_ratio: float = 0.0,
        fp_option: FingerprintOption = FingerprintOption()
    ) -> None:
        super().__init__()
        self._max_num_atoms = max_num_atoms  # ??
        # self._max_smiles_len = max_smiles_len
        self._max_num_reactions = max_num_reactions
        self._fpindex = fpindex
        self._fp_option = fp_option 
        # self._reaction_matrix = reaction_matrix
        self._protein_molecule_pairs = protein_molecule_pairs  
        self._protein_embeddings = protein_embeddings 
        self._synthetic_pathways = synthetic_pathways 
        self._init_stack_weighted_ratio = init_stack_weighted_ratio
        self._virtual_length = virtual_length

    def __len__(self) -> int:
        return self._virtual_length

    def __iter__(self):
        for smiles, protein_id in self._protein_molecule_pairs:
            if smiles in self._synthetic_pathways and protein_id in self._protein_embeddings:
                pathway = torch.tensor(self._synthetic_pathways[smiles])
                token_types = pathway[:, 0]
                rxn_indices = torch.where(pathway[:,0]==2, pathway[:,1], 0)  # extract rxn_indices; fill others with 0 
                reactant_indices = torch.where(pathway[:,0]==3, pathway[:,1], 0)  # extract reactant_indices; fill others with 0 
                reactant_fps = torch.tensor(np.array([
                    self._fpindex[reactant_idx][1]  # fpindex[reactant_idx] returns (molecule, fingerprint) tuple 
                    if reactant_idx != 0 
                    else torch.zeros(self._fp_option.dim) 
                    for reactant_idx in reactant_indices 
                ]))
                protein_embeddings = self._protein_embeddings[protein_id]
                token_padding_mask = torch.zeros_like(token_types, dtype=torch.bool)
                """
                Printing some examples and shapes from original data for reference:

                # Example:
                mol_seq_full (<synformer.chem.mol.Molecule object at 0x17e0a6e90>, 
                              <synformer.chem.mol.Molecule object at 0x17cd06fb0>, 
                              <synformer.chem.mol.Molecule object at 0x30cbe4cd0>)
                mol_idx_seq_full [128510, 43161, None]  # notice: None when it's a reaction
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
                data: "ProjectionData" = {
                    "protein_embeddings": protein_embeddings,
                    "token_types": token_types,
                    "rxn_indices": rxn_indices,
                    "reactant_fps": reactant_fps,
                    "token_padding_mask": token_padding_mask,
                }
                # print("protein_embeddings", protein_embeddings.shape)
                # print("token_types", token_types.shape)
                # print("rxn_indices", rxn_indices.shape)
                # print("reactant_fps", reactant_fps.shape)
                # raise Exception()
                yield data 
        '''
        while True:
            for stack in create_stack_step_by_step(
                self._reaction_matrix,
                max_num_reactions=self._max_num_reactions,
                max_num_atoms=self._max_num_atoms,  # ??
                init_stack_weighted_ratio=self._init_stack_weighted_ratio,
            ):
                mol_seq_full = stack.mols
                mol_idx_seq_full = stack.get_mol_idx_seq()
                rxn_seq_full = stack.rxns
                rxn_idx_seq_full = stack.get_rxn_idx_seq()
                product = random.choice(list(stack.get_top()))
                data = create_data(
                    product=product,
                    protein=protein,
                    mol_seq=mol_seq_full,
                    mol_idx_seq=mol_idx_seq_full,
                    rxn_seq=rxn_seq_full,
                    rxn_idx_seq=rxn_idx_seq_full,
                    fpindex=self._fpindex,
                )
                # data["smiles"] = data["smiles"][: self._max_smiles_len]
                yield data
        '''


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
        if not os.path.exists(self.config.chem.protein_molecule_pairs_path):
            raise FileNotFoundError(
                f"Protein-molecule pairs not found: {self.config.chem.protein_molecule_pairs_path}. "
            )
        if not os.path.exists(self.config.chem.protein_embedding_path):
            raise FileNotFoundError(
                f"Protein embeddings not found: {self.config.chem.protein_embedding_path}. "
            )

        '''
        with open(self.config.chem.rxn_matrix, "rb") as f:
            rxn_matrix = pickle.load(f)
        '''

        with open(self.config.chem.fpindex, "rb") as f:
            fpindex = pickle.load(f)
        
        with open(self.config.chem.protein_molecule_pairs_path, "rb") as f:
            protein_molecule_pairs = pd.read_csv(f).to_numpy()  # [n_proteins, 2]
            print(len(protein_molecule_pairs), "\t ", "protein-molecule pairs")
        
        with open(self.config.chem.protein_embedding_path, "rb") as f:
            # protein_embeddings = torch.load(f)  # dict {protein_id: embedding}, embedding [N, 960]
            protein_embeddings = torch.load(f, map_location=torch.device("cpu"))
            print(len(protein_embeddings), "\t ", "protein embeddings")

        with open(self.config.chem.synthetic_pathways_path, "rb") as f:
            # synthetic_pathways = torch.load(f)  # dict {smiles: [(token_type, reaction_or_reactant_index)]}  # each value [n_tokens, 2]
            synthetic_pathways = torch.load(f, map_location=torch.device("cpu"))
            print(len(synthetic_pathways), "\t ", "synthetic pathways")

        self.train_dataset = ProjectionDataset(
            # reaction_matrix=rxn_matrix,
            fpindex=fpindex,
            protein_molecule_pairs=protein_molecule_pairs,
            protein_embeddings=protein_embeddings,
            synthetic_pathways=synthetic_pathways, 
            virtual_length=self.config.train.val_freq * self.batch_size,
            **self.dataset_options, 
        )
        # TODO: once we have separate files for train/val datasets, create val_dataset
        self.val_dataset = self.train_dataset  # for now, use the same dataset for validation
        '''
        self.val_dataset = ProjectionDataset(
            # reaction_matrix=rxn_matrix,
            fpindex=fpindex,
            protein_molecule_pairs=protein_molecule_pairs,
            protein_embeddings=protein_embeddings,
            virtual_length=self.batch_size,
            **self.dataset_options, 
        )
        '''

    def train_dataloader(self):
        use_workers = self.num_workers > 0
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=Collater(),
            worker_init_fn=worker_init_fn,
            persistent_workers=use_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=Collater(),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )


# TODO: modify to enable protein embeddings and synthetic pathways
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
            reaction_matrix=rxn_matrix,
            fpindex=fpindex,
            protein_embeddings=protein_embeddings,
            virtual_length=self.config.train.val_freq * self.batch_size,
            **self.dataset_options,
        )
        self.val_dataset = ProjectionDataset(
            reaction_matrix=rxn_matrix,
            fpindex=fpindex,
            protein_embeddings=protein_embeddings,
            virtual_length=self.batch_size,
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=Collater(),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

# E.g. python -m scripts.sample_naive --model-path data/trained_weights/epoch=23-step=28076.ckpt --device cpu 
import pathlib
import pickle
import pandas as pd 

import click
import torch
from omegaconf import OmegaConf

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.mol import Molecule
from synformer.models.synformer import Synformer


def load_model(model_path: pathlib.Path, config_path: pathlib.Path | None, device: torch.device):
    ckpt = torch.load(model_path, map_location="cpu")

    if config_path is None:
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        model = Synformer(config.model).to(device)
        model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
    else:
        config = OmegaConf.load(config_path)
        model = Synformer(config.model).to(device)
        model.load_state_dict(ckpt)
    model.eval()

    fpindex: FingerprintIndex = pickle.load(open(config.chem.fpindex, "rb"))
    rxn_matrix: ReactantReactionMatrix = pickle.load(open(config.chem.rxn_matrix, "rb"))
    return model, fpindex, rxn_matrix


def featurize_smiles(smiles: str, device: torch.device, repeat: int = 1):
    mol = Molecule(smiles)
    atoms, bonds = mol.featurize_simple()
    atoms = atoms[None].repeat(repeat, 1).to(device)
    bonds = bonds[None].repeat(repeat, 1, 1).to(device)
    num_atoms = atoms.size(0)
    atom_padding_mask = torch.zeros([1, num_atoms], dtype=torch.bool, device=device)

    smiles_t = mol.tokenize_csmiles()
    smiles_t = smiles_t[None].repeat(repeat, 1).to(device)
    feat = {
        "atoms": atoms,
        "bonds": bonds,
        "atom_padding_mask": atom_padding_mask,
        "smiles": smiles_t,
    }
    return mol, feat


def load_protein_molecule_pairs(protein_molecule_pairs_val_path="data/protein_molecule_pairs/papyrus_val_19399.csv"):
    df_protein_molecule_pairs = pd.read_csv(protein_molecule_pairs_val_path)
    df_protein_molecule_pairs["short_target_id"] = df_protein_molecule_pairs["target_id"].apply(lambda s: s.split("_")[0])
    df_protein_molecule_pairs = df_protein_molecule_pairs.set_index("SMILES")
    return df_protein_molecule_pairs


def load_protein_embeddings(protein_embeddings_path="data/protein_embeddings/embeddings_selection_float16_4973.pth"):
    with open(protein_embeddings_path, "rb") as f:
        return torch.load(f, map_location=torch.device("cpu")) 


def sample(smiles, target, model, fpindex, rxn_matrix, protein_embeddings, device, repeat=1):
    mol, feat = featurize_smiles(smiles, device, repeat=repeat)
    feat["protein_embeddings"] = protein_embeddings[target].unsqueeze(0).repeat(repeat, 1, 1).float()
    with torch.inference_mode():
        result = model.generate_without_stack(
            feat,
            rxn_matrix=rxn_matrix,
            fpindex=fpindex,
            temperature_token=1.0,
            temperature_reactant=0.1,
            temperature_reaction=1.0,
        )
        ll = model.get_log_likelihood(
            code=result.code,
            code_padding_mask=result.code_padding_mask,
            token_types=result.token_types,
            rxn_indices=result.rxn_indices,
            reactant_fps=result.reactant_fps,
            token_padding_mask=result.token_padding_mask,
        )
    stacks = result.build()
    cnt = 0
    info = {}
    for i, stack in enumerate(stacks):
        if stack.get_stack_depth() == 1:
            analog = stack.get_one_top()
            cnt += 1
            info[i] = {
                "smiles": analog.smiles,
                "analog": analog,
                "stack": stack,
                "ll": ll["total"][i].sum().item(),
                "cnt_rxn": stack.count_reactions(), 
                "similarity": analog.sim(mol)
            }
            # print({k: v for k, v in info[i].items() if k in ("ll", "cnt_rxn", "similarity")})
    # print(f"Total: {cnt} / {len(stacks)}")
    return info, result 


@click.command()
@click.option("--smiles", type=str, default="O=C(Nc1nnc(C2CC2)s1)C1CC(=O)N(c2cc3c(cc2)OCCO3)C1")
@click.option("--target", type=str, default=None)  # protein ID as used in the Papyrus dataset, e.g. "Q9Y468_WT"
@click.option("--model-path", type=click.Path(exists=True, path_type=pathlib.Path), required=True)
@click.option("--config-path", type=click.Path(exists=True, path_type=pathlib.Path), required=False, default=None)
@click.option("--device", type=torch.device, default="cuda")
@click.option("--repeat", type=int, default=100)
def main(smiles, target, model_path: pathlib.Path, config_path: pathlib.Path | None, device: torch.device, repeat: int):
    model, fpindex, rxn_matrix = load_model(model_path, config_path, device)
    protein_embeddings = load_protein_embeddings()
    df_protein_molecule_pairs = load_protein_molecule_pairs()
    if target is None:
        print("Fetching a suitable protein target from the dataset...")
        target = df_protein_molecule_pairs.loc[smiles, "target_id"]
    print("SMILES:", smiles)
    print("Target:", target)
    sample(smiles, target, model, fpindex, rxn_matrix, protein_embeddings, device, repeat=repeat)


if __name__ == "__main__":
    main()

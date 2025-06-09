import pathlib
import pickle
import pandas as pd 
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


def load_protein_molecule_pairs(path="data/protein_molecule_pairs/papyrus_val_19399.csv"):
    df_protein_molecule_pairs = pd.read_csv(path)
    df_protein_molecule_pairs["short_target_id"] = df_protein_molecule_pairs["target_id"].apply(lambda s: s.split("_")[0])
    df_protein_molecule_pairs = df_protein_molecule_pairs.set_index("SMILES")
    return df_protein_molecule_pairs


def load_protein_embeddings(path="data/protein_embeddings/embeddings_selection_float16_4973.pth"):
    with open(path, "rb") as f:
        return torch.load(f, map_location=torch.device("cpu")) 


def sample(target, model, fpindex, rxn_matrix, protein_embeddings, device, true_smiles=None, repeat=1):
    if true_smiles is not None:
        mol, feat = featurize_smiles(true_smiles, device, repeat=repeat)
    else:
        feat = {}
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
            }
            if true_smiles is not None:
                info[i]["similarity"] = analog.sim(mol)
            # print({k: v for k, v in info[i].items() if k in ("ll", "cnt_rxn", "similarity")})
    # print(f"Total: {cnt} / {len(stacks)}")
    return info, result 

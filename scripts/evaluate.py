# E.g. python -m scripts.evaluate data/trained_weights/epoch=23-step=28076.ckpt --repeat 100 
from omegaconf import OmegaConf
import numpy as np 
import torch 
import click
import os 
import pickle 
import datetime as dt 
from scripts.sample_helpers import (
    load_model, 
    sample, 
    load_protein_molecule_pairs, 
    load_protein_embeddings
)


@click.command()
@click.argument("model_path", type=click.Path(exists=True))  # e.g. "data/trained_weights/epoch=23-step=28076.ckpt"
@click.option("--seed", type=int, default=42)
# @click.option("--num-workers", type=int, default=8)
# @click.option("--devices", type=int, default=1)  # default=4
# @click.option("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", 1)))
@click.option("--repeat", type=int, default=10)
@click.option("--n-examples", type=int, default=None)
def main(
    model_path: str,
    seed: int,
    # num_workers: int,
    # devices: int,
    # num_nodes: int,
    # log_dir: str,
    repeat: int,
    n_examples: int, 
):
    config = OmegaConf.load("configs/evaluation.yml")
    
    # Get model (checkpoint) name from model path
    ckpt_name = os.path.split(model_path)[-1].split(".")[0] 

    if config.system.device in ("gpu", "cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

    # Load model and data 
    model, fpindex, rxn_matrix = load_model(model_path, None, "cuda" if config.system.device == "gpu" else config.system.device)
    protein_embeddings = load_protein_embeddings()
    df_protein_molecule_pairs = load_protein_molecule_pairs(config.data.protein_molecule_pairs_path).reset_index()
    unique_target_ids = df_protein_molecule_pairs["target_id"].unique()
    # synthetic_pathways = torch.load(config.data.synthetic_pathways_path, map_location=torch.device("cpu"))

    print(len(unique_target_ids), "unique test proteins")

    if n_examples is not None:
        unique_target_ids = np.random.choice(unique_target_ids, n_examples, replace=False)

    infos, results = {}, {}
    for target_id in unique_target_ids:  
        print(f"Sampling molecules for protein {target_id}")
        info, result = sample(
            target_id, 
            model, 
            fpindex, 
            rxn_matrix, 
            protein_embeddings, 
            config.system.device, 
            repeat=repeat
        )
        # TODO: this saves all generations, even ones that do not result in a valid molecule 
        #  Probably makes sense to only keep successful ones? i.e. those that are in info's keys 
        # Serialize reactions and reactants objects inside result
        result.reactions = [
            [
                {"smarts": reaction._smarts, "num_reactions": reaction.num_reactants} or None 
                if reaction is not None else None 
                for reaction in entry
            ] 
            for entry in result.reactions
        ]
        result.reactants = [
            [
                {"smiles": reactant._smiles} 
                if reactant is not None else None 
                for reactant in entry
            ] 
            for entry in result.reactants
        ]
        # Save info
        infos[target_id] = info 
        # Save result 
        results[target_id] = result 

    # Create directory to save info/result to, if it doesn't exist yet 
    evaluations_path = "data/evaluations"
    if not os.path.exists(evaluations_path):
        print(f"Creating directory {evaluations_path}")
        os.mkdir(evaluations_path)
    ckpt_evaluations_path = os.path.join(evaluations_path, ckpt_name)
    if not os.path.exists(ckpt_evaluations_path):
        print(f"Creating directory {ckpt_evaluations_path}")
        os.mkdir(ckpt_evaluations_path)
    
    # Save to files
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    info_path = os.path.join(ckpt_evaluations_path, f"info {timestamp}.pkl")
    with open(info_path, "wb") as f:
        print(f"Saving to {info_path}")
        pickle.dump(info, f)
    result_path = os.path.join(ckpt_evaluations_path, f"result {timestamp}.pkl")
    with open(result_path, "wb") as f:
        print(f"Saving to {result_path}")
        pickle.dump(result, f)


if __name__ == "__main__":
    main()

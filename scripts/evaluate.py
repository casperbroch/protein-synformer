# Example usage:
# python -m scripts.evaluate data/trained_weights/epoch=23-step=28076.ckpt --repeat 100 

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
@click.argument("model_path", type=click.Path(exists=True))  # Path to trained model checkpoint
@click.option("--seed", type=int, default=42)                # Random seed
@click.option("--repeat", type=int, default=10)              # Number of generations per protein
@click.option("--n-examples", type=int, default=None)        # Limit number of proteins to sample from
def main(
    model_path: str,
    seed: int,
    repeat: int,
    n_examples: int, 
):
    # Load evaluation config
    config = OmegaConf.load("configs/evaluation.yml")
    
    # Extract checkpoint name from path
    ckpt_name = os.path.split(model_path)[-1].split(".")[0] 

    # Enable performance optimization if on GPU
    if config.system.device in ("gpu", "cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

    # Load model and required data
    model, fpindex, rxn_matrix = load_model(
        model_path, None, 
        "cuda" if config.system.device == "gpu" else config.system.device
    )
    protein_embeddings = load_protein_embeddings()
    df_protein_molecule_pairs = load_protein_molecule_pairs(config.data.protein_molecule_pairs_path).reset_index()
    unique_target_ids = df_protein_molecule_pairs["target_id"].unique()

    print(len(unique_target_ids), "unique test proteins")

    # If specified, sample a subset of proteins
    if n_examples is not None:
        unique_target_ids = np.random.choice(unique_target_ids, n_examples, replace=False)

    infos, results = {}, {}

    # Sample for each protein target
    for target_id in unique_target_ids:  
        print(f"Sampling molecules for protein {target_id}")
        info, result = sample(
            target_id, 
            model, 
            fpindex, 
            rxn_matrix, 
            protein_embeddings, 
            config.system.device, 
            repeat=repeat,
        )

        # Convert reaction and reactant objects to serializable dictionaries
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

        # Store results
        infos[target_id] = info 
        results[target_id] = result 

    # Create save directories if needed
    evaluations_path = "data/evaluations"
    if not os.path.exists(evaluations_path):
        print(f"Creating directory {evaluations_path}")
        os.mkdir(evaluations_path)
    ckpt_evaluations_path = os.path.join(evaluations_path, ckpt_name)
    if not os.path.exists(ckpt_evaluations_path):
        print(f"Creating directory {ckpt_evaluations_path}")
        os.mkdir(ckpt_evaluations_path)
    
    # Save info and result files with timestamp
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    info_path = os.path.join(ckpt_evaluations_path, f"infos {timestamp}.pkl")
    with open(info_path, "wb") as f:
        print(f"Saving to {info_path}")
        pickle.dump(infos, f)
    result_path = os.path.join(ckpt_evaluations_path, f"results {timestamp}.pkl")
    with open(result_path, "wb") as f:
        print(f"Saving to {result_path}")
        pickle.dump(results, f)


if __name__ == "__main__":
    main()

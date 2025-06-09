# E.g. python -m scripts.sample_naive --model-path data/trained_weights/epoch=23-step=28076.ckpt --device cpu 
import click 
from scripts.sample_helpers import * 


@click.command()
@click.option("target", type=str)  # protein ID as used in the Papyrus dataset, e.g. "Q9Y468_WT"
@click.option("--smiles", type=str, default=None) 
@click.option("--model-path", type=click.Path(exists=True, path_type=pathlib.Path), required=True)
@click.option("--config-path", type=click.Path(exists=True, path_type=pathlib.Path), required=False, default=None)
@click.option("--device", type=torch.device, default="cuda")
@click.option("--repeat", type=int, default=100)
def main(
    target: str, 
    smiles: str, 
    model_path: pathlib.Path, 
    config_path: pathlib.Path | None, 
    device: torch.device, 
    repeat: int
):
    model, fpindex, rxn_matrix = load_model(model_path, config_path, device)
    protein_embeddings = load_protein_embeddings()
    # df_protein_molecule_pairs = load_protein_molecule_pairs()
    print("Target:", target)
    print("SMILES:", smiles)
    sample(
        target, 
        model, 
        fpindex, 
        rxn_matrix, 
        protein_embeddings, 
        device, 
        true_smiles=smiles, 
        repeat=repeat,
        # temperature_token=...,
        # temperature_reactant=...,
        # temperature_reaction=...
    )


if __name__ == "__main__":
    main()

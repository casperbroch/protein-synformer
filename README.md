# Protein-Synformer

A protein‑conditioned fork of **SynFormer**
Generative modelling and synthesis‑planning for drug discovery.

---

## 1. Overview

Prot2Drug‑SynFormer extends the original **SynFormer** framework to generate *synthetic routes* for small‑molecule ligands **conditioned on a target protein**.
Key additions come from our MSc research (see paper draft in `/docs/`) and include:

| New feature                                                                                                   | Where to look                                                               |
| ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Protein‑aware decoder using per‑residue embeddings from **ESM‑C 600 M**                                       | `src/synformer/models/protein_decoder.py`                                   |
| **Updated** **`scripts/train.py`** – PyTorch‑Lightning datamodule, mixed‑precision, gradient accumulation | `scripts/train.py`                                                          |
| **`scripts/hyperopt.py`** – Optuna‑powered hyper‑parameter search                                             | `scripts/hyperopt.py` (requires [Optuna](https://github.com/optuna/optuna)) |
| **`scripts/evaluate.py`** – unified evaluation (route validity, length, novelty, docking proxy)               | `scripts/evaluate.py`                                                       |
| Pre‑computed protein embeddings (16‑bit, ESM‑C 600 M)                                                         | `data/protein_embeddings/`                                                  |


The fork retains SynFormer’s fast transformer‑based inference and support for *Enamine REAL* building blocks while adding protein context and modern training utilities.

---

## 2. Installation

### 2.1 Environment

```bash
# create conda env\conda env create -f env.yml -n prot2drug
conda activate prot2drug

# install package in editable mode
pip install --no-deps -e .
```

The provided `env.yml` includes **PyTorch 2.3**, **PyTorch‑Lightning 2.2**, **Optuna 3.x**, and RDKit nightly.

### 2.2 Building Block Database

We provide pre‑processed building‑block data (`fpindex.pkl` and `matrix.pkl`). You can download these files from [our Hugging Face page](https://huggingface.co/whgao/synformer) and place them in the `data` directory specified in your YAML config.

**Important:** the files are derived from Enamine’s *REAL* building‑block catalogue and are **available only upon request**. Please request the <ins>US Stock</ins> catalogue from Enamine and store the raw SMILES under `data/building_blocks`. The processed data are for academic research only; any commercial use requires permission from Enamine.

### 2.3 Trained Models

Pre‑trained checkpoints can be downloaded from the same Hugging Face page and stored in `data/trained_weights/`.

### 2.4 Train with your own templates and building blocks

If you prefer a custom set of reaction templates or building blocks, edit the paths on lines 6–7 of the relevant config file and run:

```bash
python scripts/preprocess.py --model-config your_config_file.yml
```

to regenerate `fpindex.pkl` and `matrix.pkl`, then train:

```bash
python scripts/train.py configs/dev_smiles_diffusion.yml
```

Tweak the batch size and number of epochs according to your hardware.


### 2.2 Data assets

| Asset                                                       | Location                                    | Notes                                                                                                                |
|-------------------------------------------------------------|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| **Building blocks**                                         | `data/building_blocks/`                    | Raw Enamine REAL US-in-stock catalogue. Request access via [Enamine](https://enamine.net).                         |
| **Processed block data** (`fpindex.pkl`, `matrix.pkl`)      | `data/processed/comp_2048/`                | Fingerprint index and reaction template matrix, derived from building blocks.                                       |
| **Protein embeddings** (`*.pth`)                            | `data/protein_embeddings/`                 | ESM-C 600 M per-residue embeddings stored in 16-bit PyTorch format.                                                 |
| **Synthetic routes** (`*.pth`)                              | `data/synthetic_pathways/`                 | Pickled synthetic pathways from SynFormer runs; used for supervised pretraining or evaluation.                      |
| **Protein–molecule pairs** (`*.csv`)   | `data/protein_molecule_pairs/`                          | Tabular file containing known binding pairs for training and evaluation.                                            |
| **Trained checkpoints**                                     | `data/trained_weights/`                    | Pretrained model weights: both *small* (~17 M) and *big* (~178 M) model variants available.                         |

> **Note**: For access to the synthetic pathways, protein embeddings, trained weights and protein molecule pairs, please contact the maintainers of this repository directly.

Download links are available on our [Hugging Face hub](https://huggingface.co/prot2drug/synformer).

---

## 3. Quick start

### 3.1 Training

```bash
python scripts/train.py configs/prot2drug.yml \
  --batch-size 128 \
  --devices 1 \
  --num-workers 8 \
  --log-dir ./logs
```

Key flags:

* **`--debug`** – runs a single forward & backward pass to sanity‑check the pipeline.
* **`--resume <ckpt>`** – warms‑start the **decoder** and **head** weights from a checkpoint (see in‑script comments for fine‑tuning logic with LoRA or last‑N‑layers freeze).
* **`--seed`**, **`--num-nodes`**, **`--num-sanity-val-steps`** – exposed for reproducibility & multi‑node training.

> **Tip 💡** `batch_size` **must** be divisible by `devices`. 

---

### 3.2 Hyper‑parameter optimisation

```bash
python scripts/hyperopt.py configs/prot2drug_big.yml \
  --n-trials 100 \
  --batch-size 196 \
  --devices 1 \
  --n-jobs 4        # run four Optuna trials in parallel
```

---

### 3.3 Inference

```bash
python sample.py \
  --model-path checkpoints/prot2drug_big.ckpt \
  --protein-fasta P12345.fasta \
  --output results/P12345_routes.csv
```

By default the script searches `data/protein_embeddings/` for a pre‑computed embedding; otherwise it will generate one on‑the‑fly (GPU recommended). Sampling parameters (beam width, temperatures, max steps) can be modified via CLI – run `sample.py --help`.

---

### 3.4 Evaluation

```bash
python scripts/evaluate.py data/trained_weights/epoch=23-step=28076.ckpt \
  --repeat 100 \
  --n-examples 50
```

---

## 4. Citation

If you use Prot2Drug‑SynFormer in your research, please cite:

```bibtex
@mastersthesis{euskirchen2025prot2drug,
  title = {x},
  author = {Janik Euskirchen, Gerben Prikanowski, Casper Bröcheler, Humam Ahmed},
  school = {Maastricht University},
  year = {2025}
}

@article{gao2024synformer,
  title={Generative Artificial Intelligence for Navigating Synthesizable Chemical Space},
  author={Gao, Wenhao and Luo, Shitong and Coley, Connor W.},
  year={2024},
  note={arXiv:2410.03494}
}
```

---

## 5. License

This fork inherits the MIT licence from SynFormer.
Use of the Enamine REAL building blocks remains subject to Enamine’s terms.

---

*Happy synthesizing!*

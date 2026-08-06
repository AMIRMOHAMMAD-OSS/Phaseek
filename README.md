# Phaseek: LLPS Prediction and Phase-Separating Peptide Generation

## Overview

Phaseek is a sequence-based computational tool for studying protein liquid–liquid phase separation (LLPS).

Phaseek provides two main capabilities:

1. **Prediction**  
   Scores amino-acid sequences for their predicted LLPS propensity and identifies residue-level regions associated with phase separation.

2. **Peptide generation**  
   Designs de novo amino-acid sequences with high predicted LLPS propensity using a SeqProp-inspired differentiable optimization method.

Phaseek supports:

- Phaseek v1 and Phaseek v2
- single-sequence prediction
- batch prediction from FASTA files
- residue-level LLPS profiles
- long-sequence inference using overlapping windows
- gradient-based peptide generation
- Google Colab execution

## Online Resources

- [Run Phaseek in Google Colab](https://colab.research.google.com/github/AMIRMOHAMMAD-OSS/Phaseek/blob/main/phaseek.ipynb)
- [Phaseek model files on Hugging Face](https://huggingface.co/AmirMMH/Phaseek)
- [Phaseek preprint on bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.27.635039)

## Model Architecture

![Phaseek architecture](Picture11.svg)

Phaseek combines contextual protein-sequence representations with statistical protein graph features to predict LLPS propensity.

Phaseek v2 uses a Transformer-based sequence encoder together with FEGS graph matrices. The model produces a global sequence-level LLPS score, while the prediction pipeline also provides residue-level LLPS scores.

## Installation

Clone the repository:

```bash
git clone https://github.com/AMIRMOHAMMAD-OSS/Phaseek.git
cd Phaseek
```

Creating a virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Phaseek v2 checkpoint

For local Phaseek v2 inference, the trained checkpoint must be available at:

```text
Functions/Phaseek_v2/model/best.pt
```

The Google Colab notebook downloads and configures the required model files automatically.

## Command-Line Usage

The main prediction interface is:

```text
Functions/runner.py
```

### Single-sequence prediction with Phaseek v2

```bash
python Functions/runner.py \
  --model v2 \
  --sequence "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGYGSGRGKGGKGLGGKGLGKGGAKRHRK" \
  --id test_sequence \
  --directory test_results \
  --device auto \
  --window-size 512 \
  --stride 256 \
  --aggregation mean
```

### Single-sequence prediction with Phaseek v1

```bash
python Functions/runner.py \
  --model v1 \
  --sequence "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQ" \
  --id test_sequence_v1 \
  --directory test_results
```

### FASTA batch prediction

```bash
python Functions/runner.py \
  --model v2 \
  --fasta proteins.fasta \
  --id FASTA_results \
  --directory batch_results \
  --end-sequence 100 \
  --device auto \
  --window-size 512 \
  --stride 256 \
  --aggregation mean
```

The `--end-sequence` argument limits how many records are read from the FASTA file. It does not truncate individual protein sequences.

## Main CLI Arguments

| Argument | Description |
|---|---|
| `--model` | Selects `v1` or `v2`. |
| `--sequence` | Amino-acid sequence for single-sequence prediction. |
| `--fasta` | Path to a FASTA file for batch prediction. |
| `--id` | Identifier used for the result folder and output files. |
| `--directory` | Parent directory name inside `Functions/Results`. |
| `--end-sequence` | Maximum number of FASTA records to process. |
| `--device` | Selects `auto`, `cpu`, or `cuda`. |
| `--window-size` | Maximum v2 inference-window length. Default: 512. |
| `--stride` | Number of residues between consecutive v2 windows. Default: 256. |
| `--aggregation` | Combines v2 window scores using `mean` or `max`. |
| `--checkpoint` | Optional custom path to the Phaseek v2 checkpoint. |
| `--m-mat` | Optional custom path to the FEGS `M.mat` file. |

## Long Sequences

Phaseek v2 has a maximum model input length of 512 amino acids.

Sequences longer than the selected window size are divided into overlapping windows. By default:

```text
window size = 512
stride = 256
aggregation = mean
```

The last window is anchored to the end of the sequence so that the complete sequence is covered.

The final raw v2 score is calculated from the individual window scores using either mean or maximum aggregation.

## Output Files

Results are saved under:

```text
Functions/Results/<directory>/<sequence_id>/
```

For a single sequence, the output may include:

```text
scores.csv
prediction.json
window_scores.csv
```

### `scores.csv`

Contains the residue-level LLPS profile:

```text
scores
seq
```

### `prediction.json`

Contains the prediction summary, including:

- sequence identifier
- selected model
- sequence length
- final Phaseek score
- raw model score
- model threshold
- predicted class
- number of sequence windows
- window aggregation method

### `window_scores.csv`

Created for Phaseek v2 when window-level results are available. It contains:

- window number
- start position
- end position
- window score

For FASTA batch prediction, the combined output is saved as:

```text
LLPS_prediction_of_seqs.csv
```

## Peptide Generation

The Colab notebook includes a SeqProp-inspired method for generating amino-acid sequences with high predicted LLPS propensity.

The candidate sequence is represented by a trainable logit matrix:

\[
\theta \in \mathbb{R}^{L \times 20}
\]

where \(L\) is the user-defined sequence length.

The first residue is fixed to methionine. The remaining positions are optimized using:

- temperature-controlled amino-acid probabilities
- annealed Gumbel noise
- expected amino-acid embeddings
- a frozen Phaseek scoring model
- entropy regularization
- Adam optimization
- final argmax decoding

An optional local hill-climbing step can test single-residue substitutions and retain substitutions that further improve the predicted LLPS score.

For Phaseek v2, FEGS matrices are periodically reconstructed from the current argmax sequence and used as fixed graph-conditioning information between refresh steps.

Generated sequences are saved in both CSV and FASTA formats.

## Key Features

- Phaseek v1 and Phaseek v2 prediction
- sequence-level LLPS scoring
- residue-level LLPS profiling
- single-sequence and FASTA batch inference
- overlapping-window analysis for long sequences
- configurable mean or maximum window aggregation
- SeqProp-inspired peptide generation
- optional discrete sequence refinement
- CSV, JSON, FASTA, SVG, and structure-related outputs
- Google Colab workflow

## Issues and Contributions

Bug reports and feature requests may be submitted through GitHub Issues.

Code contributions, modified versions, and redistribution are governed by the Phaseek license. Before distributing or submitting a Contribution, users must contact Inserm Transfert to determine whether and under which conditions the Contribution may be distributed.

## License

Phaseek v2.0 is distributed under the:

**Phaseek License – research purposes restricted**

The complete license is available in:

[LICENSE.pdf](LICENSE.pdf)

Phaseek v2.0, including its source code, object code, trained models, model weights, user interfaces, graphical interfaces, and documentation, may be used only for strictly academic and non-commercial research under the conditions of the license.

Uses that are not authorized under this research license include, among others:

- clinical trials
- diagnostic or therapeutic use
- fee-based research services
- projects or research collaborations with for-profit organizations
- use in a commercial product
- commercial or industrial product and process development
- other activities intended to obtain a commercial advantage or financial compensation

Any use outside the licensed scope requires a separate written agreement with Inserm Transfert.

Downloading or using Phaseek constitutes acknowledgment and acceptance of the license conditions.

### Patent notice

The peptide-generation method implemented in Phaseek is associated with patent application:

```text
EP26300824.5
COMPUTER IMPLEMENTED METHOD OF GENERATING
LIQUID-LIQUID PHASE SEPARATION PEPTIDES
```

The Phaseek research software license does not grant any rights under the associated patent.

### Licensing contacts

For questions about permitted use, commercial use, contributions, or licensing, contact Inserm Transfert:

```text
contrats_IT@inserm-transfert.fr
licensing@inserm-transfert.fr
jur@inserm-transfert.fr
propriete@inserm-transfert.fr
```

## Citation

When using Phaseek in academic research, please cite:

```bibtex
@article{MohammadHosseini_2025,
  title     = {Generalizable prediction of liquid-liquid phase separation from protein sequence},
  author    = {MohammadHosseini, Amir M. and Teimouri, Hossein and Gureghian, Vincent and Najjar, Rayan and Lindner, Ariel B. and Pandi, Amir},
  journal   = {bioRxiv},
  year      = {2025},
  month     = jan,
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.1101/2025.01.27.635039},
  url       = {https://www.biorxiv.org/content/10.1101/2025.01.27.635039},
  note      = {Preprint}
}
```

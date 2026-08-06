<p align="center">
  <img src="Picture10.svg" alt="Phaseek logo" width="650">
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/AMIRMOHAMMAD-OSS/Phaseek/blob/main/phaseek_colab.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
  <a href="https://huggingface.co/AmirMMH/Phaseek">
    <img src="https://img.shields.io/badge/Hugging%20Face-Phaseek-FFD21E" alt="Hugging Face">
  </a>
  <a href="https://www.biorxiv.org/content/10.1101/2025.01.27.635039v2">
    <img src="https://img.shields.biorxiv.org/content/10.1101/2025.01.27.635039v2">
    <img src="https://.io/badge/bioRxiv-Preprint-B31B1B" alt="bioRxiv preprint">
  </a>
  <a href="LICENSE.pdf">
    <img src="https://img.shields.io/badge/License-Research%20Purposes%20Restricted-168ACD" alt="License">
  </a>
</p>

<p align="center">
  <strong>LLPS prediction and phase-separating peptide generation</strong>
</p>

---

## Overview

Phaseek is a sequence-based computational tool for studying protein liquid–liquid phase separation (LLPS).

It provides two main functions:

- **LLPS prediction:** scores protein sequences and generates residue-level LLPS profiles.
- **Peptide generation:** designs amino-acid sequences with high predicted LLPS propensity using gradient-based sequence optimization.

Phaseek supports:

- Phaseek v1 and Phaseek v2
- single-sequence prediction
- batch prediction from FASTA files
- residue-level LLPS profiles
- overlapping-window analysis for long sequences
- SeqProp-inspired peptide generation
- optional local sequence refinement
- Google Colab execution

## Links

- [Run Phaseek in Google Colab](https://colab.research.google.com/github/AMIRMOHAMMAD-OSS/Phaseek/blob/main/phaseek_colab.ipynb)
- [Phaseek on Hugging Face](https://huggingface.co/AmirMMH/Phaseek)
- [Phaseek preprint on bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.27.635039v2)

## Model Architecture

<p align="center">
  <img src="Picture11.svg" alt="Phaseek model architecture" width="900">
</p>

Phaseek v2 combines a Transformer-based sequence encoder with FEGS graph matrices to calculate a sequence-level LLPS score.

The prediction workflow also reports a residue-level LLPS profile and a final Phaseek score.

## Installation

Clone the repository:

```bash
git clone https://github.com/AMIRMOHAMMAD-OSS/Phaseek.git
cd Phaseek
```

Create a virtual environment:

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

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Required Files

Phaseek v1 model files are stored in:

```text
model/
```

The Phaseek v2 checkpoint must be available at:

```text
Functions/Phaseek_v2/model/best.pt
```

The FEGS matrix file must be available at:

```text
Functions/M.mat
```

The Colab notebook downloads and configures the required files automatically.

## Prediction

The main command-line interface is:

```text
Functions/runner.py
```

### Phaseek v2

```bash
python Functions/runner.py \
  --model v2 \
  --sequence "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQ" \
  --id example_v2 \
  --directory example_results \
  --device auto \
  --window-size 512 \
  --stride 256 \
  --aggregation mean
```

### Phaseek v1

```bash
python Functions/runner.py \
  --model v1 \
  --sequence "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQ" \
  --id example_v1 \
  --directory example_results
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

`--end-sequence` limits the number of FASTA records processed. It does not truncate individual protein sequences.

## CLI Arguments

| Argument | Description |
|---|---|
| `--model` | Selects `v1` or `v2`. |
| `--sequence` | Amino-acid sequence for single-sequence prediction. |
| `--fasta` | Path to a FASTA file. |
| `--id` | Identifier used for the result folder. |
| `--directory` | Parent directory created inside `Functions/Results`. |
| `--end-sequence` | Maximum number of FASTA records to process. |
| `--device` | Selects `auto`, `cpu`, or `cuda`. |
| `--window-size` | Phaseek v2 inference-window length. |
| `--stride` | Distance between consecutive windows. |
| `--aggregation` | Combines window scores using `mean` or `max`. |
| `--checkpoint` | Optional path to a Phaseek v2 checkpoint. |
| `--m-mat` | Optional path to the FEGS matrix file. |

## Long Sequences

Phaseek v2 processes up to 512 residues per model call.

Longer sequences are divided into overlapping windows. The default settings are:

```text
Window size: 512
Stride: 256
Aggregation: mean
```

The last window is aligned with the end of the sequence.

Window scores are combined using either:

- `mean`
- `max`

## Output Files

Results are saved under:

```text
Functions/Results/<directory>/<sequence_id>/
```

A single-sequence prediction may produce:

```text
scores.csv
prediction.json
window_scores.csv
```

### `scores.csv`

Contains the residue-level LLPS profile:

| Column | Description |
|---|---|
| `scores` | Residue-level LLPS score |
| `seq` | Amino-acid residue |

### `prediction.json`

Contains the sequence identifier, selected model, sequence length, final score, raw model score, thresholds and window information.

### `window_scores.csv`

For Phaseek v2, this file contains:

| Column | Description |
|---|---|
| `window` | Window number |
| `start` | First residue position |
| `end` | Last residue position |
| `score` | Raw Phaseek v2 window score |

For FASTA input, the combined table is saved as:

```text
LLPS_prediction_of_seqs.csv
```

## Colab Workflow

The Colab notebook includes sections for:

1. installing Phaseek and its dependencies
2. selecting Phaseek v1 or v2
3. scoring a sequence or FASTA file
4. plotting residue-level LLPS scores
5. mapping LLPS scores onto a protein structure
6. generating phase-separating peptides
7. downloading the results

Open the notebook here:

[Phaseek Colab notebook](https://colab.research.google.com/github/AMIRMOHAMMAD-OSS/Phaseek/blob/main/phaseek_colab.ipynb)

## Phase-Separating Peptide Generation

Phaseek includes a SeqProp-inspired gradient-based method for generating sequences with high predicted LLPS propensity.

A candidate sequence is represented as a trainable logit matrix:

$$
\theta \in \mathbb{R}^{L \times 20}
$$

Here, $L$ is the sequence length selected by the user.

The first position is fixed to methionine. The remaining positions are randomly initialized.

During optimization, the logits are converted into differentiable amino-acid probabilities:

$$
P_{i,a}
=
\frac{
\exp\left((\theta_{i,a}+\alpha_t G_{i,a})/\tau_t\right)
}{
\sum_{b=1}^{20}
\exp\left((\theta_{i,b}+\alpha_t G_{i,b})/\tau_t\right)
}
$$

where:

- $G_{i,a}$ is Gumbel noise
- $\tau_t$ is the temperature at step $t$
- $\alpha_t$ is the Gumbel-noise scale

The probability matrix is projected into the model embedding space:

$$
E = P W_{AA}
$$

where $W_{AA}$ is the frozen amino-acid embedding matrix.

The sequence logits are optimized using:

$$
\mathcal{L}
=
-S_{\mathrm{LLPS}}
+
\lambda H(P)
$$

where:

- $S_{\mathrm{LLPS}}$ is the differentiable model score
- $H(P)$ is the entropy of the amino-acid probabilities
- $\lambda$ is the entropy coefficient

Only the sequence logits are updated. The Phaseek model parameters remain frozen.

After optimization, the sequence is decoded using:

$$
s_i
=
\operatorname*{argmax}_{a}
\theta_{i,a}
$$

An optional refinement step can test single-residue substitutions and retain substitutions that increase the selected model's raw score.

### Phaseek v1 Generation

For Phaseek v1, the relaxed amino-acid embeddings are passed through the frozen v1 Transformer classifier.

### Phaseek v2 Generation

For Phaseek v2, the relaxed embeddings are passed through the v2 Transformer together with FEGS graph matrices.

Because FEGS extraction requires a discrete sequence, the graph matrices are periodically reconstructed from the current argmax sequence and remain fixed between refresh steps.

The final sequences are rescored using the standard Phaseek prediction workflow.

## Generation Parameters

| Parameter | Default | Description |
|---|---:|---|
| `MODEL_VERSION` | `v2` | Selects Phaseek v1 or v2. |
| `SEQUENCE_LENGTH` | `120` | Length of each generated sequence. |
| `NUMBER_OF_SEQUENCES` | `3` | Number of independent optimization runs. |
| `RANDOM_SEED` | `42` | Controls initialization and Gumbel noise. Consecutive runs use consecutive seeds. |
| `GRADIENT_STEPS` | `500` | Number of Adam optimization steps. |
| `LEARNING_RATE` | `0.1` | Adam learning rate for the sequence logits. |
| `ENTROPY_WEIGHT` | `0.001` | Weight of the entropy penalty. |
| `TEMP_START` | `2.0` | Initial relaxation temperature. |
| `TEMP_END` | `0.1` | Final relaxation temperature. |
| `GUMBEL_NOISE_START` | `1.0` | Initial Gumbel-noise scale. |
| `V2_GRAPH_REFRESH` | `10` | Number of steps between Phaseek v2 FEGS updates. |
| `LOG_EVERY` | `50` | Progress-display interval. |
| `ENABLE_REFINEMENT` | `False` | Enables local substitution refinement. |
| `REFINEMENT_PASSES` | `1` | Maximum number of refinement passes. |

### Temperature Annealing

The temperature is decreased linearly:

$$
\tau_t
=
\tau_{\mathrm{start}}
+
\frac{t}{N-1}
\left(
\tau_{\mathrm{end}}
-
\tau_{\mathrm{start}}
\right)
$$

### Gumbel-Noise Annealing

The Gumbel-noise scale is decreased toward zero:

$$
\alpha_t
=
\alpha_{\mathrm{start}}
\left(
1-\frac{t}{N-1}
\right)
$$

### Random Seeds

With:

```text
RANDOM_SEED = 42
NUMBER_OF_SEQUENCES = 3
```

the runs use:

```text
42
43
44
```

### Local Refinement

When refinement is enabled, positions after the fixed methionine are tested with alternative amino acids.

One refinement pass may require up to:

$$
19(L-1)
$$

additional model evaluations.

## Generated Peptide Files

Generated results are saved under the selected generation directory. The default location is:

```text
Functions/Results/Generated_peptides/
```

The main files are:

```text
generated_peptides.csv
generated_peptides.fasta
generation_settings.json
```

Each generated sequence also receives its own Phaseek result folder.

## Repository Structure

```text
Phaseek/
├── Functions/
│   ├── Phaseek_v1/
│   ├── Phaseek_v2/
│   ├── XG_Boost/
│   ├── phaseek_v2/
│   ├── FEGS_feature_extraction.py
│   ├── M.mat
│   ├── runner.py
│   └── Results/
├── model/
├── phaseek_colab.ipynb
├── Picture10.svg
├── Picture11.svg
├── LICENSE.pdf
├── README.md
└── requirements.txt
```

## Issues and Contributions

Bug reports and feature requests can be submitted through GitHub Issues.

Contributions, modified versions and redistribution are subject to the Phaseek license. Users should contact Inserm Transfert before distributing a contribution or modified version.

## License

Phaseek v2.0 is distributed under the:

**Phaseek License – research purposes restricted**

The complete license is available in:

[LICENSE.pdf](LICENSE.pdf)

Use is restricted to academic and non-commercial research under the full terms of the license.

Uses outside the permitted scope require a separate agreement with Inserm Transfert.

The software license does not grant rights under the associated patent.

### Licensing Contacts

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
  url       = {https://www.biorxiv.org/content/10.1101/2025.01.27.635039v2},
  note      = {Preprint}
}
```

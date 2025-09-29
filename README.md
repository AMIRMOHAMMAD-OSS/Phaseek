<p align="center">
  <img src="Picture10.svg" alt="Phaseek Banner" width="100%" height="100%" style="vertical-align: middle;"/>
</p>

<p align="center">
  <a href="https://www.biorxiv.org/content/10.1101/2025.01.27.635039v2" target="_blank"><img src="https://img.shields.io/badge/bioRxiv-Preprint-2C974B?style=for-the-badge&logo=readthedocs&logoColor=white" alt="bioRxiv"></a>
  <a href="https://github.com/AMIRMOHAMMAD-OSS/Phaseek"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://huggingface.co/AmirMMH/Phaseek"><img src="https://img.shields.io/badge/HuggingFace-Model-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Model"></a>
  <a href="https://colab.research.google.com/github/AMIRMOHAMMAD-OSS/Phaseek/blob/main/phaseek.ipynb"><img src="https://img.shields.io/badge/Colab-Notebook-e2006a?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"></a>
</p>

# Phaseek: Liquid–Liquid Phase Separation (LLPS) Protein Scorer

## Overview
**Phaseek** predicts the propensity of protein sequences to undergo **liquid–liquid phase separation (LLPS)**. It supports single-sequence scoring and batch inference from FASTA files, with options to target specific sequence ranges via the `End_Sequence` parameter.

- Try it in the cloud: **[Google Colab](https://colab.research.google.com/github/AMIRMOHAMMAD-OSS/Phaseek/blob/main/phaseek.ipynb)**  
- Model on 🤗: **[HuggingFace](https://huggingface.co/AmirMMH/Phaseek)**  
- Paper (preprint): **[bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.27.635039v2)**

## Abstract
Liquid–liquid phase separation (LLPS) underlies the formation of biomolecular condensates and is strongly influenced by protein sequence features, particularly within intrinsically disordered regions. **Phaseek** provides a lightweight, scalable predictor that scores the LLPS propensity of protein sequences. It enables rapid screening of single sequences or entire FASTA datasets while remaining easy to deploy (Colab-ready) and integrate into pipelines.

## Installation

```bash
git clone https://github.com/AMIRMOHAMMAD-OSS/Phaseek.git
cd Phaseek

# (Optional) create a virtual environment
pip install virtualenv
virtualenv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### CLI Example
```bash
python Functions/runner.py \
  --sequence "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGYGSGRGKGGKGLGGKGLGKGGAKRHRK" \
  --id test_sequence \
  --directory test_results
```

### Notes
- **Batch mode:** Supply a FASTA file to process multiple sequences (≤ ~100 recommended for speed).  
- **Targeted analysis:** Provide a sequence ID or directory and use `End_Sequence` to truncate scoring at a given position.  
- **Model file path:** If running from `Functions/`, make sure the model path in `XGBoost.py` points to `../model/xgb_model.pkl`.

## Key Features
- 🔬 **Single or batch scoring** from plain sequences or FASTA files  
- 🎯 **Custom range** analysis via `End_Sequence`  
- ⚡ **Lightweight & fast**; Colab support for zero-setup runs  
- 🔗 **Ecosystem ready** with a HuggingFace model for integration

## Contribution
Contributions are welcome! Please fork the repository and open a pull request. For larger changes, open an issue first to discuss your proposal.

## Citation
If you use **Phaseek** in your research, please cite the preprint:

```bibtex
@article{MohammadHosseini_2025,
  title   = {Generalizable prediction of liquid–liquid phase separation from protein sequence},
  author  = {MohammadHosseini, Amir M. and Teimouri, Hossein and Gureghian, Vincent and Najjar, Rayan and Lindner, Ariel B. and Pandi, Amir},
  journal = {bioRxiv},
  year    = {2025},
  month   = jan,
  publisher = {Cold Spring Harbor Laboratory},
  doi     = {10.1101/2025.01.27.635039},
  url     = {https://www.biorxiv.org/content/10.1101/2025.01.27.635039v1},
  note    = {Preprint}
}
```

# Phaseek: Liquid–Liquid Phase Separation (LLPS) Protein Scorer

<p align="center">
  <img src="Picture10.svg" alt="Phaseek Logo" width="600"/>
</p>

<p align="center">
  <a href="https://www.biorxiv.org/content/10.1101/2025.01.27.635039v2"><img src="https://img.shields.io/badge/bioRxiv-Preprint-green?style=for-the-badge" alt="bioRxiv"></a>
  <a href="https://github.com/AMIRMOHAMMAD-OSS/Phaseek"><img src="https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://huggingface.co/AmirMMH/Phaseek"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"></a>
  <a href="https://colab.research.google.com/github/AMIRMOHAMMAD-OSS/Phaseek/blob/main/phaseek.ipynb"><img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"></a>
</p>

---

## 📖 Overview

**Phaseek** is a lightweight and generalizable tool for predicting the propensity of protein sequences to undergo **liquid–liquid phase separation (LLPS)**.  

By combining machine learning with protein sequence features, Phaseek enables rapid and scalable scoring of single proteins or large FASTA datasets.  

It was developed to support research in **biomolecular condensates**, **intrinsically disordered regions**, and LLPS-related mechanisms in biology.

---

## ✨ Features

- 🔬 **Single-sequence mode**: Quickly score one protein sequence.  
- 📂 **Batch mode**: Upload a FASTA file (up to ~100 sequences recommended).  
- 🎯 **Targeted analysis**: Specify sequence IDs or directories for scoring.  
- 🧩 **Custom range**: Use the `End_Sequence` parameter to restrict analysis.  
- ☁️ **Colab support**: Run directly in the cloud without setup.  
- 🤗 **HuggingFace model**: Available for easy integration with other ML pipelines.  

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AMIRMOHAMMAD-OSS/Phaseek.git ~/Phaseek
cd ~/Phaseek
```

### 2. Set Up a Virtual Environment
```bash
# Install virtualenv if not already available
pip install virtualenv

# Create and activate environment
virtualenv .venv
source .venv/bin/activate   # Linux / macOS
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the main runner script:

```bash
python Functions/runner.py --sequence "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGYGSGRGKGGKGLGGKGLGKGGAKRHRK" --id test_sequence --directory test_results
```

This will:
- Score the provided sequence  
- Assign the ID `test_sequence`  
- Save results in the `test_results` directory  

---

## 💻 Google Colab

No local installation? Try Phaseek instantly on [Google Colab](https://colab.research.google.com/github/AMIRMOHAMMAD-OSS/Phaseek/blob/main/phaseek.ipynb).  

---

## 🧑‍🔬 Citation

If you use **Phaseek** in your research, please cite the associated preprint:

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

---

## 🔗 Resources
- 📄 [bioRxiv Preprint](https://www.biorxiv.org/content/10.1101/2025.01.27.635039v2)  
- 🤗 [HuggingFace Model Hub](https://huggingface.co/AmirMMH/Phaseek)  
- 💻 [Google Colab Notebook](https://colab.research.google.com/github/AMIRMOHAMMAD-OSS/Phaseek/blob/main/phaseek.ipynb)

---

✨ Phase separation is at the frontier of cell biology — Phaseek aims to make it easier for everyone to explore the sequence determinants behind it.

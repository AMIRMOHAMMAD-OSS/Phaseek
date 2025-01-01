import os
import sys
import time
import math
import json
import pickle
import random
import shutil
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from ast import literal_eval

# Bioinformatics libraries
from Bio import SeqIO, pairwise2
from Bio.Seq import Seq

# Tokenization and Transformers
from tokenizers import Tokenizer
from transformers import AutoTokenizer, CanineTokenizer, CanineModel

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.preprocessing import StandardScaler as Sc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as tts
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

# Utility imports
from command_runner import command_runner

# Load tokenizer
tokenizer = Tokenizer.from_file("classi/Trained_BPE2.json")
tokenizer.model_max_length = 256

# Load XGBoost model
with open('classi/xgb_model (1).pkl', 'rb') as f:
    clf = pickle.load(f)

# Prepare datasets
def edit(sequence):
    """Removes unwanted characters from sequences."""
    for char in "BJOUX\Z_n\n":
        sequence = sequence.replace(char, "")
    return sequence

def encode(sequence):
    """Encodes a sequence into its token IDs."""
    vocab = tokenizer.get_vocab()
    return [vocab[char] for char in sequence]

def Padding(sequences, PAD=0, max_len=512):
    """Pads sequences to the same length."""
    max_len = max(512, len(max(sequences, key=len)))
    return np.array([
        seq + [PAD] * (max_len - len(seq)) if len(seq) < max_len else seq 
        for seq in sequences
    ])

# Load training, validation, and negative datasets
with open("classi/training.txt", "r", encoding="utf-8") as f:
    train_text = f.read()
filtered_train = [seq for seq in edit(train_text).split("<|edoftext|>") if seq]

with open("classi/validation.txt", "r", encoding="utf-8") as f:
    val_text = f.read()
filtered_val = [seq for seq in edit(val_text).split("<|edoftext|>") if seq]

with open("classi/negative_dataset.txt", "r") as f:
    neg_text = f.read()
filtered_neg = [seq for seq in neg_text.split("\n") if seq]

# Combine positive datasets and write to a FASTA file
pos_sequences = list(set(filtered_train + filtered_val))
with open("DrLLPS.fasta", "w") as fasta_file:
    for idx, seq in enumerate(pos_sequences):
        fasta_file.write(f">seq{idx}\n{seq}\n")

print("Environment setup complete! ✨🍰✨")

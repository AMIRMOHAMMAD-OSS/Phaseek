#!/usr/bin/env python3
import argparse,hashlib
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
p=argparse.ArgumentParser(); p.add_argument('--root',default='.'); p.add_argument('--m-mat',default='assets/M.mat'); a=p.parse_args(); root=Path(a.root).resolve()
df=pd.read_csv(root/'data/manifest_sequences_baseline.csv'); expected={('train',1):593,('train',0):4167,('val',1):66,('val',0):370}; actual=df.groupby(['split','label']).size().to_dict()
if actual!=expected: raise RuntimeError(actual)
if df.sample_id.duplicated().any(): raise RuntimeError('duplicate sample IDs')
m=root/a.m_mat
if not m.is_file(): raise FileNotFoundError(f'Missing {m}; copy $WORK/Phaseek/Functions/M.mat')
mat=loadmat(m)
if 'M' not in mat: raise RuntimeError('M missing from M.mat')
h=hashlib.sha256(m.read_bytes()).hexdigest(); print('counts',actual); print('motifs',len(mat['M'].flatten())); print('M.mat sha256',h); print('OK')

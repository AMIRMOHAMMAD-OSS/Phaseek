#!/usr/bin/env python3
import argparse,pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--topk',type=int,default=10); a=p.parse_args(); df=pd.read_csv(a.manifest)
for split,g in df.groupby('split'):
 c=(a.topk*(g.length.astype(int)**2)).sum(); print(f'{split}: {len(g)} | float16={c*2/1024**3:.3f} GiB | float32={c*4/1024**3:.3f} GiB')
c=(a.topk*(df.length.astype(int)**2)).sum(); print(f'total: {len(df)} | float16={c*2/1024**3:.3f} GiB | float32={c*4/1024**3:.3f} GiB')

"""
File: check-results.py
Author: Chuncheng Zhang
Date: 2026-04-27
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Check results.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-04-27 ------------------------
# Requirements and constants
from pathlib import Path
import pandas as pd

# %%
DATA_DIR = Path('./results')


# %% ---- 2026-04-27 ------------------------
# Function and class



# %% ---- 2026-04-27 ------------------------
# Play ground
files = sorted(DATA_DIR.rglob('training_log.csv'))
table = []
for file in files:
    df = pd.read_csv(file)
    mode, subj, _ = file.parent.name.split('-', 2)
    print(f'File: {file}')
    auc = df['auc'].max()
    # auc = df['auc'].iloc[-1]
    print(f'Mode: {mode}, Subject: {subj}, AUC: {auc:.4f}')
    table.append({'mode': mode, 'subject': subj, 'auc': auc})
    # print(df.head())

table = pd.DataFrame(table)
print(table)


# %% ---- 2026-04-27 ------------------------
# Pending



# %% ---- 2026-04-27 ------------------------
# Pending

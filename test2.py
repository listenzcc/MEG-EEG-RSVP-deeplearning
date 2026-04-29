# %%
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pathlib import Path
import mne
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
# ---------------------------
# Temporal + Spatial Embedding
# ---------------------------


class EEGPatchEmbedding(nn.Module):
    def __init__(self, in_ch=64, emb_dim=128, kernel_size=25):
        super().__init__()

        # Temporal conv
        self.temporal = nn.Conv2d(
            1, emb_dim,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            bias=False
        )

        # Spatial projection (across channels)
        self.spatial = nn.Conv2d(
            emb_dim, emb_dim,
            kernel_size=(in_ch, 1),
            bias=False
        )

        self.bn = nn.BatchNorm2d(emb_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, C, T]
        x = x.unsqueeze(1)          # [B, 1, C, T]
        x = self.temporal(x)        # [B, D, C, T]
        x = self.spatial(x)         # [B, D, 1, T]
        x = self.bn(x)
        x = self.act(x)

        x = x.squeeze(2)            # [B, D, T]
        x = x.permute(0, 2, 1)      # [B, T, D]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim=128, num_heads=4, depth=4, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

    def forward(self, x):
        # x: [B, T, D]
        return self.encoder(x)


class AttentionPooling(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, emb_dim))

    def forward(self, x):
        # x: [B, T, D]
        B = x.size(0)
        q = self.query.expand(B, -1, -1)  # [B, 1, D]

        attn = torch.matmul(q, x.transpose(1, 2))  # [B, 1, T]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, x)  # [B, 1, D]
        return out.squeeze(1)        # [B, D]


class RSVPTransformer(nn.Module):
    def __init__(
        self,
        in_ch=64,
        emb_dim=128,
        depth=4,
        num_heads=4,
        num_classes=1
    ):
        super().__init__()

        self.embedding = EEGPatchEmbedding(
            in_ch=in_ch,
            emb_dim=emb_dim
        )

        self.transformer = TransformerEncoder(
            emb_dim=emb_dim,
            num_heads=num_heads,
            depth=depth
        )

        self.pool = AttentionPooling(emb_dim)

        self.cls = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, C, T]
        x = self.embedding(x)       # [B, T, D]
        x = self.transformer(x)     # [B, T, D]
        x = self.pool(x)            # [B, D]
        x = self.cls(x)             # [B, 1]
        # activation is included in loss function (e.g., BCEWithLogitsLoss)
        x = torch.sigmoid(x)            # [B, 1]
        x = x.squeeze(1)             # [B]
        return x


def aggregate_trials(logits):
    # logits: [B, N_trials, 1]
    return logits.mean(dim=1)


class TrialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, N, D]
        w = self.fc(x)                  # [B, N, 1]
        w = torch.softmax(w, dim=1)
        return (x * w).sum(dim=1)


def focal_loss(logits, targets, alpha=0.25, gamma=2):
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    return loss.mean()


# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='EEG')
parser.add_argument('--subj', type=str, default='S02')
parser.add_argument('--device', type=int, default=5)
options = parser.parse_args()
print(options)

MODE = options.mode
SUBJ = options.subj
device = options.device

print(f'Working on {MODE} - {SUBJ}')

# %%

DATA_DIR = Path(f'decoding-step-1/{MODE}-{SUBJ}')
assert DATA_DIR.exists(), f'{DATA_DIR} does not exist'

from datetime import datetime
timestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = Path(f'results/{MODE}3-{SUBJ}-{timestr}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%

def read_epochs(path, label_bin, label_tri):
    epochs = mne.read_epochs(path, preload=True)
    epochs.apply_baseline()

    ch_names = epochs.ch_names
    print(f'{path.name}: {len(epochs)} epochs, channels: {ch_names}', file=open(OUTPUT_DIR / 'log.txt', 'a'))

    # todo: for MEG, only pick O, P, T, C channels (occipital, parietal, temporal, central), which are more relevant for RSVP. For EEG, keep all channels.
    # 272 -> 204
    if MODE == 'MEG' and True:
        epochs.pick([e for e in ch_names if e[2] in ['O', 'P', 'T', 'C']])

    # epochs.filter(l_freq=0.1, h_freq=40, method='iir', n_jobs=-1)
    epochs.crop(0, 1)
    # X shape is (n_epochs, n_channels, n_times)
    X = epochs.get_data()

    # todo: detrend timeseries in order 1

    X = X[:, :, :200]
    y_bin = np.full((X.shape[0],), label_bin)
    y_tri = np.full((X.shape[0],), label_tri)
    return X, y_bin, y_tri


# %%
X1, y1_bin, y1_tri = read_epochs(
    DATA_DIR / 'epochs-1-epo.fif', label_bin=1, label_tri=1)
X2, y2_bin, y2_tri = read_epochs(
    DATA_DIR / 'epochs-2-epo.fif', label_bin=0, label_tri=0)
# X3, y3_bin, y3_tri = read_epochs(
#     DATA_DIR / 'epochs-3-epo.fif', label_bin=0, label_tri=2)

# X = np.concatenate((X1, X2, X3), axis=0)
# y = np.concatenate((y1_bin, y2_bin, y3_bin), axis=0)

X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1_bin, y2_bin), axis=0)
# y_tri = np.concatenate((y1_tri, y2_tri, y3_tri), axis=0)
print(X.shape, y.shape)

# exit(0)
# %%

# 3000 trials, each with 64 channels and 256 time points
# X_train = torch.randn(3000, in_ch, 200).cuda(device=device)
# y_train = torch.randint(0, 2, (3000, 1)).float().cuda(device=device)

# X_val = torch.randn(500, in_ch, 200).cuda(device=device)
# y_val = torch.randint(0, 2, (500, 1)).float().cuda(device=device)

n_samples, in_ch, n_times = X.shape

cv = StratifiedKFold(n_splits=5, shuffle=True)
for train_idx, test_idx in cv.split(X, y):
    # print(train_idx, test_idx)
    break

X_train, X_val = X[train_idx], X[test_idx]
y_train, y_val = y[train_idx], y[test_idx]


print(f'{X_train.shape=}, {y_train.shape=}')
print(f'{X_val.shape=}, {y_val.shape=}')


# example output:
# X.shape: (n_trials, n_channels, n_times) = (2876, 62, 201)
# y.shape: (n_trials,) = (2876,)

# %%
X_train = torch.from_numpy(X_train).float().cuda(device=device)
y_train = torch.from_numpy(y_train).float().cuda(device=device)
X_val = torch.from_numpy(X_val).float().cuda(device=device)
y_val = torch.from_numpy(y_val).float().cuda(device=device)
# y_train = y_train.unsqueeze(1)  # [B, 1]
# y_val = y_val.unsqueeze(1)      # [B, 1]


def normalize(x):
    # return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)


X_train = normalize(X_train)
X_val = normalize(X_val)

# %%
# default: in_ch=64, emb_dim=128, depth=4, num_heads=4, num_classes=1
# emb_dim: 64, 128, 256, 512, MEG seems lower emb_dim works better, maybe because of the small dataset size
# emb_dim should be divisible by num_heads
# todo: num_heads = 8
# test in 2.2
model = RSVPTransformer(in_ch=in_ch).cuda(device=device)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# criterion = focal_loss

optimizer = torch.optim.AdamW(
    model.parameters(),
    # lr=3e-4,        # ↓↓↓
    lr=1e-3,        # ↓↓↓
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50
)
pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# %%
# Train with train set, validate with validation set
# Using torch util board to log training and validation loss

batch_size = 64
batch_size = 128

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=batch_size,
    shuffle=False
)

# %%

writer = SummaryWriter(log_dir=OUTPUT_DIR / 'logs')

num_epochs = 50
best_val_loss = float('inf')

results = []

for epoch in tqdm(range(num_epochs), desc="Epochs"):

    # -------------------
    # Train
    # -------------------
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb = xb.cuda(device=device)
        yb = yb.cuda(device=device)

        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        # loss.backward()
        # optimizer.step()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    # -------------------
    # Validation
    # -------------------
    model.eval()
    val_loss = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.cuda(device=device)
            yb = yb.cuda(device=device)

            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss += loss.item() * xb.size(0)

            all_logits.append(logits)
            all_labels.append(yb)

    val_loss /= len(val_loader.dataset)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    probs = torch.sigmoid(all_logits)

    # -------------------
    # Metrics（RSVP重点）
    # -------------------
    from sklearn.metrics import roc_auc_score

    try:
        auc = roc_auc_score(
            all_labels.cpu().numpy(),
            probs.cpu().numpy()
        )
    except:
        auc = 0.0

    # -------------------
    # Logging
    # -------------------
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("AUC/val", auc, epoch)
    results.append({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'auc': auc
    })

    print(
        f"[{epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, AUC={auc:.4f}")

    # -------------------
    # Early stopping
    # -------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")

import pandas as pd
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "training_log.csv", index=False)

# %%

# %%

# %%

# %%

# %%

import gzip, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NUC2IDX = {"A":0, "C":1, "G":2, "T":3}

def extract_3x5mers(seq):
    if isinstance(seq, str) and len(seq) >= 7:
        return [seq[0:5], seq[1:6], seq[2:7]]
    return [None, None, None]

def encode_3x5(seq_list):
    return torch.tensor([[NUC2IDX.get(b, 0) for b in kmer] for kmer in seq_list], dtype=torch.long)

def normalize_reads(reads, mean_vals, std_vals):
    arr = np.array(reads, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 9 or len(arr) == 0:
        arr = np.zeros((1, 9), dtype=np.float32)
    return (arr - mean_vals) / (std_vals + 1e-6)

def pad_reads(reads_np):
    if reads_np.ndim != 2 or reads_np.shape[1] != 9 or len(reads_np) == 0:
        reads_np = np.zeros((1, 9), dtype=np.float32)
    return torch.tensor(reads_np, dtype=torch.float32)

def read_train_data(json_gz_path, labels_csv_path):
    """Training: read gz JSON and merge with labels."""
    with gzip.open(json_gz_path, "rb") as f:
        lines = f.read().decode().strip().split("\n")
    rec = []
    for line in lines:
        obj = json.loads(line)
        for tid, vals in obj.items():
            for pos, sub in vals.items():
                for nuc, reads in sub.items():
                    rec.append([tid, pos, nuc, reads])
    df = pd.DataFrame(rec, columns=["transcript_id", "transcript_position", "nucleotide", "reads"])
    df["transcript_position"] = df["transcript_position"].astype(int)
    labels = pd.read_csv(labels_csv_path)
    df = pd.merge(labels, df, on=["transcript_id", "transcript_position"])
    df["5mers"] = df["nucleotide"].apply(extract_3x5mers)
    return df

def read_pred_data(json_gz_path):
    """Prediction: read gz JSON without labels."""
    with gzip.open(json_gz_path, "rb") as f:
        lines = f.read().decode().strip().split("\n")
    rec = []
    for line in lines:
        obj = json.loads(line)
        for tid, vals in obj.items():
            for pos, sub in vals.items():
                for nuc, reads in sub.items():
                    rec.append([tid, pos, nuc, reads])
    df = pd.DataFrame(rec, columns=["transcript_id", "transcript_position", "nucleotide", "reads"])
    df["transcript_position"] = df["transcript_position"].astype(int)
    df["5mers"] = df["nucleotide"].apply(extract_3x5mers)
    return df

class TrainDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.mean_vals = np.vstack(df["reads"]).mean(axis=0)
        self.std_vals = np.vstack(df["reads"]).std(axis=0)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        reads = pad_reads(normalize_reads(row["reads"], self.mean_vals, self.std_vals))
        seq = encode_3x5(row["5mers"])
        y = torch.tensor([float(row["label"])], dtype=torch.float32)
        return reads, seq, y

class PredDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.mean_vals = np.vstack(df["reads"]).mean(axis=0)
        self.std_vals = np.vstack(df["reads"]).std(axis=0)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        reads = pad_reads(normalize_reads(row["reads"], self.mean_vals, self.std_vals))
        seq = encode_3x5(row["5mers"])
        return reads, seq

def collate_train(batch):
    reads_list, seqs_list, labels_list = zip(*batch)
    max_len = max(r.size(0) for r in reads_list)
    padded_reads, masks = [], []
    for r in reads_list:
        R = r.size(0)
        if R < max_len:
            r = torch.cat([r, torch.zeros((max_len - R, r.size(1)), dtype=r.dtype)], dim=0)
        mask = torch.zeros(max_len, dtype=torch.bool); mask[:R] = True
        padded_reads.append(r); masks.append(mask)
    return torch.stack(padded_reads), torch.stack(seqs_list), torch.stack(labels_list), torch.stack(masks)

def collate_pred(batch):
    reads_list, seqs_list = zip(*batch)
    max_len = max(r.size(0) for r in reads_list)
    padded_reads, masks = [], []
    for r in reads_list:
        R = r.size(0)
        if R < max_len:
            r = torch.cat([r, torch.zeros((max_len - R, r.size(1)), dtype=r.dtype)], dim=0)
        mask = torch.zeros(max_len, dtype=torch.bool); mask[:R] = True
        padded_reads.append(r); masks.append(mask)
    return torch.stack(padded_reads), torch.stack(seqs_list), torch.stack(masks)

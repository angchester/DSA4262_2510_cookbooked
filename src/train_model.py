import os, numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from src.data_utils import read_train_data, TrainDataset, collate_train
from src.model_arch import SiteClassifier, FocalLoss

# Example call for train script -> python main.py train <unlabelled data> <data_labels>
# For dataset0 (provided on canvas and add into /data folder) call -> python main.py train dataset0.json.gz data.info.labelled
# WARNING: The expected training time for this model is roughly 14 hours on a g4dn.4xlarge GPU machine. For your convenience, the final trained model files can be found in /model of the repo

@torch.no_grad()
def infer_logits(model, dl):
    model.eval()
    all_logits, all_labels = [], []
    for reads, seq, y, mask in dl:
        logits, _ = model(reads, seq, mask)
        all_logits.extend(logits.squeeze().cpu().tolist())
        all_labels.extend(y.squeeze().cpu().tolist())
    return np.array(all_logits), np.array(all_labels)


def train_one_epoch(model, dl, opt, loss_fn):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dl, desc="Training", leave=False)
    for reads, seq, y, mask in pbar:
        opt.zero_grad()
        logits, _ = model(reads, seq, mask)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / max(1, len(dl))


def train_bagged_models(
    dataset_name: str,
    label_name: str,
    base_dir="data",
    save_dir="models",
    num_bags=5,
    num_folds=10,
    epochs=5,
    batch_size=32,
    lr=1e-4
):
    """Train bagged models using dataset and label files inside /data directory."""

    # Dynamically construct full paths (no hardcoding)
    data_path = os.path.join(base_dir, dataset_name)
    label_path = os.path.join(base_dir, label_name)

    # Input validation
    if not isinstance(dataset_name, str) or not isinstance(label_name, str):
        raise TypeError("Invalid input: expected file names as strings.")

    if not (dataset_name.endswith(".json") or dataset_name.endswith(".json.gz")):
        raise ValueError(f"Invalid dataset format: {dataset_name} (expected .json or .json.gz)")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at {label_path}")

    # Setup
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Dataset: {data_path}")
    print(f"Labels : {label_path}")
    print(f"Output : {save_dir}/")
    print(f"Device : {device}")

    # Load data
    df = read_train_data(data_path, label_path)
    pos, neg = (df["label"] == 1).sum(), (df["label"] == 0).sum()
    alpha = neg / (pos + neg)
    gamma = 1.25
    print(f"α={alpha:.4f}, γ={gamma:.2f}")

    # Bagging
    SEEDS = [4262 + i for i in range(num_bags)]
    for bag_idx, seed in enumerate(SEEDS, 1):
        print(f"\n=== Bag {bag_idx}/{num_bags} | SEED={seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = SiteClassifier().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)

        gkf = GroupKFold(n_splits=num_folds)
        all_fold_metrics = []

        for fold, (tr, va) in enumerate(gkf.split(df, groups=df["gene_id"]), 1):
            print(f"\nFold {fold}/{num_folds}")
            train_dl = DataLoader(TrainDataset(df.iloc[tr]), batch_size=batch_size, shuffle=True, collate_fn=collate_train)
            val_dl = DataLoader(TrainDataset(df.iloc[va]), batch_size=batch_size, shuffle=False, collate_fn=collate_train)

            for ep in range(epochs):
                loss = train_one_epoch(model, train_dl, opt, loss_fn)
                print(f"Epoch {ep+1}: Loss={loss:.4f}")

            logits, labels = infer_logits(model, val_dl)
            preds = torch.sigmoid(torch.tensor(logits)).numpy()
            metrics = {
                "fold": fold,
                "acc": accuracy_score(labels, preds > 0.5),
                "f1": f1_score(labels, preds > 0.5)
            }
            all_fold_metrics.append(metrics)

        df_metrics = pd.DataFrame(all_fold_metrics)
        print(f"\n{num_folds}-Fold Summary (Bag {bag_idx}):\n", df_metrics.mean(numeric_only=True).round(4))

        model_path = os.path.join(save_dir, f"bag{bag_idx}_finalmodel.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model → {model_path}")

    print("\nAll bagged models trained and saved successfully.")
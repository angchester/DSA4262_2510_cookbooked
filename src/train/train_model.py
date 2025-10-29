import os, numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data_utils import read_train_data, TrainDataset, collate_train
from src.model_arch import SiteClassifier, FocalLoss


@torch.no_grad()
def infer_logits(model, dl):
    """Evaluate model on a validation DataLoader."""
    model.eval()
    all_logits, all_labels = [], []
    for reads, seq, y, mask in dl:
        logits, _ = model(reads, seq, mask)
        all_logits.extend(logits.squeeze().cpu().tolist())
        all_labels.extend(y.squeeze().cpu().tolist())
    return np.array(all_logits), np.array(all_labels)


def train_one_epoch(model, dl, opt, loss_fn):
    """Train for one epoch with progress bar."""
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
    data_path="data/dataset0_test.json.gz",
    label_path="data/data.info_test.labelled",
    save_dir="models",
    num_bags=5,
    num_folds=10,
    epochs=5,
    batch_size=32,
    lr=1e-4
):
    # 0️⃣ Check dataset existence
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "dataset0_test.json.gz is not found.\n"
            "Please download the file from the data folder in the repository."
        )
    if not os.path.exists(label_path):
        raise FileNotFoundError(
            "data.info_test.labelled is not found.\n"
            "Please download the file from the data folder in the repository."
        )

    # 1️⃣ Create save directory
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cpu")

    # 2️⃣ Load data
    df = read_train_data(data_path, label_path)
    pos, neg = (df["label"] == 1).sum(), (df["label"] == 0).sum()
    alpha = neg / (pos + neg)
    gamma = 1.25
    print(f"α={alpha:.4f}, γ={gamma:.2f}, device={device}")

    # 3️⃣ Set seeds for reproducibility
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

        # 4️⃣ Cross-validation training
        for fold, (tr, va) in enumerate(gkf.split(df, groups=df["gene_id"]), 1):
            print(f"\nFold {fold}/{num_folds}")
            train_dl = DataLoader(TrainDataset(df.iloc[tr]), batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_train)
            val_dl = DataLoader(TrainDataset(df.iloc[va]), batch_size=batch_size,
                                shuffle=False, collate_fn=collate_train)

            for ep in range(epochs):
                loss = train_one_epoch(model, train_dl, opt, loss_fn)
                print(f"Epoch {ep+1}: Loss={loss:.4f}")

            # Validation per fold
            logits, labels = infer_logits(model, val_dl)
            preds = torch.sigmoid(torch.tensor(logits)).numpy()
            metrics = {
                "fold": fold,
                "acc": accuracy_score(labels, preds > 0.5),
                "f1": f1_score(labels, preds > 0.5)
            }
            all_fold_metrics.append(metrics)

        # 5️⃣ Save bag model
        df_metrics = pd.DataFrame(all_fold_metrics)
        print(f"\n{num_folds}-Fold Summary (Bag {bag_idx}):\n",
              df_metrics.mean(numeric_only=True).round(4))

        model_path = os.path.join(save_dir, f"bag{bag_idx}_finalmodel.pt")
        torch.save(model.state_dict(), model_path)
        print(f" Saved model → {model_path}")

    print("\n All bagged models trained and saved successfully in the 'models/' folder.")

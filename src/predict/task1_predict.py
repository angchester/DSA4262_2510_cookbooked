import os, json, torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from src.data_utils import read_pred_data, PredDataset, collate_pred
from src.model_arch import SiteClassifier


# Helper: Load model 
def _load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # Rename legacy keys
    rename_map = {"seq_emb.": "seq_enc.", "att_pool.": "pool.", "site_head.": "head."}
    remapped = {}
    for k, v in state_dict.items():
        for old, new in rename_map.items():
            if k.startswith(old):
                k = k.replace(old, new)
        remapped[k] = v

    model = SiteClassifier()
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"Loaded {os.path.basename(path)} ({len(missing)} missing / {len(unexpected)} unexpected keys)")
    model.eval()
    return model


# Helper: Compute metrics and plot curves
def evaluate_from_probs(name, labels, probs, out_dir, thr=0.5):
    preds = (probs > thr).astype(int)
    metrics = {
        "acc": accuracy_score(labels, preds),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc_roc": roc_auc_score(labels, probs),
        "auc_pr": average_precision_score(labels, probs),
    }

    print(f"\n=== {name} Metrics ===")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}, AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"Precision: {metrics['prec']:.4f}, Recall: {metrics['rec']:.4f}, "
          f"F1: {metrics['f1']:.4f}, Accuracy: {metrics['acc']:.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={metrics['auc_roc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.join(out_dir, f"{name}_roc.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot PR curve 
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure()
    plt.plot(recall, precision, label=f"AUC={metrics['auc_pr']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    pr_path = os.path.join(out_dir, f"{name}_pr.png")
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC → {roc_path}")
    print(f"Saved PR  → {pr_path}")
    return metrics


# Main inference function
@torch.no_grad()
def run_task1_prediction(model_dir="models", data_dir="data", num_bags=5):
    """Run bagged ensemble inference, evaluate dataset0, and save plots."""
    device = torch.device("cpu")
    save_dir = "predictions"
    os.makedirs(save_dir, exist_ok=True)

    # Load bagged models
    model_paths = [os.path.join(model_dir, f"bag{i}_finalmodel.pt") for i in range(1, num_bags + 1)]
    models = [_load_model(p) for p in model_paths if os.path.exists(p)]
    print(f"\n Loaded {len(models)} models from '{model_dir}'")

    # Loop through dataset0 (labelled) and dataset1 (unlabelled)
    for i in range(2):
        data_path = os.path.join(data_dir, f"dataset{i}_test.json.gz")
        if not os.path.exists(data_path):
            print(f" Missing {data_path}, skipping…")
            continue

        print(f"\n Running predictions on {data_path}")
        df = read_pred_data(data_path)
        dl = DataLoader(PredDataset(df), batch_size=16, shuffle=False, collate_fn=collate_pred)

        # Predict with ensemble
        all_probs = []
        for midx, model in enumerate(models, 1):
            probs = []
            for reads, seq, mask in tqdm(dl, desc=f"Bag {midx} - dataset{i}", leave=False):
                logits, _ = model(reads, seq, mask)
                probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            all_probs.append(probs)

        ensemble_probs = np.mean(np.array(all_probs), axis=0)

        # Save predictions CSV
        out_csv = os.path.join(save_dir, f"task1_dataset{i}_pred.csv")
        pd.DataFrame({
            "transcript_id": df["transcript_id"],
            "transcript_position": df["transcript_position"],
            "score": ensemble_probs
        }).to_csv(out_csv, index=False)
        print(f" Saved predictions → {out_csv}")

        # Evaluate and plot metrics (dataset0 only)
        if i == 0:
            label_path = os.path.join(data_dir, "data.info_test.labelled")
            if os.path.exists(label_path):
                labels = pd.read_csv(label_path)
                merged = pd.merge(labels, pd.read_csv(out_csv),
                                  on=["transcript_id", "transcript_position"])
                metrics = evaluate_from_probs("dataset0", merged["label"].values,
                                              merged["score"].values, save_dir)

                out_json = os.path.join(save_dir, "task1_dataset0_metrics.json")
                with open(out_json, "w") as f:
                    json.dump({"metrics": metrics}, f, indent=4)
                print(f" Saved metrics JSON → {out_json}")
            else:
                print(f" Missing label file: {label_path}")

    print("\n Task 1 predictions completed")

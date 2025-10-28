import os, gc, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_utils import read_pred_data, PredDataset, collate_pred
from src.model_arch import SiteClassifier

def _load_model(path, device):
    m = SiteClassifier().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

@torch.no_grad()
def run_task2_prediction(model_dir="models", task2_root="task2_data", num_bags=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = [os.path.join(model_dir, f"bag{i}_finalmodel.pt") for i in range(1, num_bags + 1)]
    models = [_load_model(p, device) for p in model_paths if os.path.exists(p)]
    print(f"Loaded {len(models)} models.")

    if not os.path.isdir(task2_root):
        print(f" {task2_root} not found"); return

    folders = [f for f in os.listdir(task2_root) if os.path.isdir(os.path.join(task2_root, f))]
    for folder in folders:
        data_path = os.path.join(task2_root, folder, "data.json")
        if not os.path.exists(data_path):
            print(f" {data_path} missing, skip"); continue

        df = read_pred_data(data_path)
        dl = DataLoader(PredDataset(df), batch_size=16, shuffle=False, collate_fn=collate_pred)

        all_probs = []
        for midx, m in enumerate(models, 1):
            probs = []
            for reads, seq, mask in tqdm(dl, desc=f"Bag {midx} - {folder}"):
                reads, seq, mask = reads.to(device), seq.to(device), mask.to(device)
                logits, _ = m(reads, seq, mask)
                probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            all_probs.append(probs)

        ensemble = np.mean(np.array(all_probs), axis=0)
        out_csv = os.path.join(model_dir, f"task2_pred_{folder}.csv")
        pd.DataFrame({
            "transcript_id": df["transcript_id"],
            "transcript_position": df["transcript_position"],
            "score": ensemble
        }).to_csv(out_csv, index=False)
        print(f" saved {out_csv}")

        del df, dl, all_probs, ensemble
        gc.collect(); torch.cuda.empty_cache()
import os, gc, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_utils import read_pred_data, PredDataset, collate_pred
from src.model_arch import SiteClassifier

def _load_model(path, device):
    m = SiteClassifier().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

@torch.no_grad()
def run_task2_prediction(model_dir="models", task2_root="task2_data", num_bags=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = [os.path.join(model_dir, f"bag{i}_finalmodel.pt") for i in range(1, num_bags + 1)]
    models = [_load_model(p, device) for p in model_paths if os.path.exists(p)]
    print(f"Loaded {len(models)} models.")

    if not os.path.isdir(task2_root):
        print(f" {task2_root} not found"); return

    folders = [f for f in os.listdir(task2_root) if os.path.isdir(os.path.join(task2_root, f))]
    for folder in folders:
        data_path = os.path.join(task2_root, folder, "data.json")
        if not os.path.exists(data_path):
            print(f" {data_path} missing, skip"); continue

        df = read_pred_data(data_path)
        dl = DataLoader(PredDataset(df), batch_size=16, shuffle=False, collate_fn=collate_pred)

        all_probs = []
        for midx, m in enumerate(models, 1):
            probs = []
            for reads, seq, mask in tqdm(dl, desc=f"Bag {midx} - {folder}"):
                reads, seq, mask = reads.to(device), seq.to(device), mask.to(device)
                logits, _ = m(reads, seq, mask)
                probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            all_probs.append(probs)

        ensemble = np.mean(np.array(all_probs), axis=0)
        out_csv = os.path.join(model_dir, f"task2_pred_{folder}.csv")
        pd.DataFrame({
            "transcript_id": df["transcript_id"],
            "transcript_position": df["transcript_position"],
            "score": ensemble
        }).to_csv(out_csv, index=False)
        print(f" saved {out_csv}")

        del df, dl, all_probs, ensemble
        gc.collect(); torch.cuda.empty_cache()

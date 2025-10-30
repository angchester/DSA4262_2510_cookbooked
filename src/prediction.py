import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_utils import read_pred_data, PredDataset, collate_pred
from src.model_arch import SiteClassifier


# --------------------------------------------------
# Helper: Load model 
# --------------------------------------------------
def _load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # Rename legacy keys for compatibility
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


# --------------------------------------------------
# Main inference function
# --------------------------------------------------
@torch.no_grad()
def run_prediction(input_file, model_dir="models", data_dir="data", num_bags=5):
    """
    Run bagged ensemble inference on a given .json or .json.gz file and save probabilities as CSV.

    Args:
        input_file (str): Filename of the data file inside data_dir.
        model_dir (str): Directory containing trained bagged models.
        data_dir (str): Directory containing the input data file.
        num_bags (int): Number of bagged models to ensemble.
    """
    save_dir = "predictions"
    os.makedirs(save_dir, exist_ok=True)

    # --- Load bagged models ---
    model_paths = [os.path.join(model_dir, f"bag{i}_finalmodel.pt") for i in range(1, num_bags + 1)]
    models = [_load_model(p) for p in model_paths if os.path.exists(p)]
    print(f"Loaded {len(models)} model(s) from '{model_dir}'")

    # --- Validate input file ---
    data_path = os.path.join(data_dir, input_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Error: {input_file} not found in '{data_dir}'. "
            "Please ensure the file exists before running predictions."
        )

    print(f"Running predictions on {data_path}")
    df = read_pred_data(data_path)
    dl = DataLoader(PredDataset(df), batch_size=16, shuffle=False, collate_fn=collate_pred)

    # --- Predict with ensemble ---
    all_probs = []
    for midx, model in enumerate(models, 1):
        probs = []
        for reads, seq, mask in tqdm(dl, desc=f"Bag {midx}", leave=False):
            logits, _ = model(reads, seq, mask)
            probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        all_probs.append(probs)

    ensemble_probs = np.mean(np.array(all_probs), axis=0)

    base_name = os.path.basename(input_file)
    if base_name.endswith(".json.gz"):
        base_name = base_name[:-8]
    elif base_name.endswith(".json"):
        base_name = base_name[:-5]

    out_csv = os.path.join(save_dir, f"{base_name}_pred.csv")

    pd.DataFrame({
        "transcript_id": df["transcript_id"],
        "transcript_position": df["transcript_position"],
        "score": ensemble_probs
    }).to_csv(out_csv, index=False)

    print(f"Saved predictions → {out_csv}")

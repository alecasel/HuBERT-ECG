import os, numpy as np, torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.serialization import add_safe_globals
from hubert_ecg import HuBERTECG, HuBERTECGConfig
from dataset import ECGDataset

CKPT    = r"F:\models\hubert_ecg_small_hf_init.pt"
ECGDIR  = r"F:\records\records_npy_raw"
TRAIN   = r"C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG\reproducibility\sph\sph_train_small.csv"
VAL     = r"C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG\reproducibility\sph\sph_val_small.csv"
OUTDIR  = r"F:\models\probes"
BATCH   = 32
DOWN    = 5

add_safe_globals([HuBERTECGConfig])
torch.set_num_threads(1)  # piÃ¹ stabile su Windows/CPU

def make_loader(csv_path, label_start_index=4):
    ds = ECGDataset(
        path_to_dataset_csv=csv_path,
        ecg_dir_path=ECGDIR,
        label_start_index=label_start_index,
        downsampling_factor=DOWN,
        pretrain=False,
        random_crop=False
    )
    dl = DataLoader(
        ds, batch_size=BATCH, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False,
        collate_fn=ds.collate
    )
    return ds, dl

def extract_embeddings(model, dl):
    X, Y = [], []
    model.eval()
    with torch.no_grad():
        for ecg, _, labels in dl:
            out = model(ecg, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True)
            H = out['last_hidden_state']         # [BS, T, D]
            pooled = H.mean(dim=1).cpu().numpy() # [BS, D]
            X.append(pooled)
            Y.append(labels.squeeze().numpy())
    return np.vstack(X), np.vstack(Y)

def main():
    # 1) Carica encoder HuBERT-ECG
    try:
        ckpt = torch.load(CKPT, map_location="cpu")
    except TypeError:
        ckpt = torch.load(CKPT, map_location="cpu")
    hubert = HuBERTECG(ckpt['model_config'])
    hubert.load_state_dict(ckpt['model_state_dict'], strict=False)

    # 2) Estrai embedding
    train_ds, train_dl = make_loader(TRAIN, label_start_index=4)
    val_ds,   val_dl   = make_loader(VAL,   label_start_index=4)

    Xtr, Ytr = extract_embeddings(hubert, train_dl)
    Xva, Yva = extract_embeddings(hubert, val_dl)

    os.makedirs(OUTDIR, exist_ok=True)
    np.save(os.path.join(OUTDIR, "X_train.npy"), Xtr)
    np.save(os.path.join(OUTDIR, "Y_train.npy"), Ytr)
    np.save(os.path.join(OUTDIR, "X_val.npy"),   Xva)
    np.save(os.path.join(OUTDIR, "Y_val.npy"),   Yva)

    # 3) Linear probe
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, verbose=0)
    )
    clf.fit(Xtr_s, Ytr)

    # 4) Valutazione
    P = clf.predict_proba(Xva_s)
    Yhat = (P >= 0.5).astype(int)

    f1_mac   = f1_score(Yva, Yhat, average="macro", zero_division=0)
    prec_mac = precision_score(Yva, Yhat, average="macro", zero_division=0)
    rec_mac  = recall_score(Yva, Yhat, average="macro", zero_division=0)
    print(f"Linear probe â€” F1 macro: {f1_mac:.4f} | Precision macro: {prec_mac:.4f} | Recall macro: {rec_mac:.4f}")

    # 5) Salva predizioni con intestazioni
    cols = train_ds.diagnoses_cols
    pd.DataFrame(Yhat, columns=cols).to_csv(os.path.join(OUTDIR, "preds_val_linear_probe.csv"), index=False)
    print("Predizioni salvate in preds_val_linear_probe.csv")

if __name__ == "__main__":
    main()

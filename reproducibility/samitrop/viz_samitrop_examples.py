# -*- coding: utf-8 -*-
import sys
import os

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importa il preprocessing del progetto (resample->500Hz + band-pass + scaling)
# richiede il fix in utils.py: usare scipy.signal.resample invece di sklearn.utils.resample
# in: (12, L) @orig_fs -> out: (12, L') @500Hz
from code2.utils import ecg_preprocessing

# ========= PERCORSI (adegua se hai messo i file altrove) =========
CSV_PATH = r"F:\SaMi-Trop dataset\exams.csv"
H5_PATH = r"F:\SaMi-Trop dataset\exams.hdf5"
OUT_DIR = r"C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG\reproducibility\samitrop\figs"

ORIG_FS = 400.0  # Hz degli ECG in exams.hdf5
LEADS = ['DI', 'DII', 'DIII', 'aVR', 'aVL',
         'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def read_csv_robusto(path):
    # alcuni rilasci hanno encoding latin1 o BOM: tenta in cascata
    for enc in ("utf-8-sig", "latin1", "ISO-8859-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except UnicodeDecodeError:
            continue
    # ultimo tentativo grezzo
    return pd.read_csv(path, encoding_errors="ignore")


def pick_indices(df: pd.DataFrame):
    # normalizza la colonna 'death' in booleano robusto
    if df["death"].dtype != bool:
        df["death"] = df["death"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False})
    i_false = df.index[df["death"] == False]
    i_true = df.index[df["death"] == True]
    if len(i_false) == 0 or len(i_true) == 0:
        raise RuntimeError(
            "Non trovo almeno un esempio per death==False e uno per death==True")
    return int(i_false[0]), int(i_true[0])


def preprocess_trace(x_4096x12: np.ndarray):
    """x (4096, 12) @400Hz -> (12, L_500Hz) preprocessato come nel repo."""
    x = x_4096x12.astype(np.float32).T  # (12, 4096)
    x = ecg_preprocessing(x, original_frequency=ORIG_FS)  # (12, L) @500Hz
    return x


def plot_12leads(ecg_12xL: np.ndarray, title: str, out_png: str, seconds: float = 5.0):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fs = 500.0
    L = min(ecg_12xL.shape[1], int(seconds*fs))
    t = np.arange(L) / fs

    fig, axes = plt.subplots(6, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()
    for k in range(12):
        axes[k].plot(t, ecg_12xL[k, :L], linewidth=0.9)
        axes[k].set_title(LEADS[k], fontsize=9)
        axes[k].grid(True, alpha=0.3)
    axes[-2].set_xlabel("Time (s)")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_leadII(ecg_12xL: np.ndarray, title: str, out_png: str, seconds: float = 5.0):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fs = 500.0
    L = min(ecg_12xL.shape[1], int(seconds*fs))
    t = np.arange(L) / fs
    y = ecg_12xL[1, :L]  # DII

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, y, linewidth=1.0)
    ax.set_title(title + " — Lead II")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("a.u.")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    # 1) leggi CSV in modo robusto
    df = read_csv_robusto(CSV_PATH)

    # 2) apri l'HDF5 e prendi il dataset "tracings"
    with h5py.File(H5_PATH, "r") as f:
        assert "tracings" in f, "Nel file HDF5 non esiste un dataset 'tracings'."
        tracings = f["tracings"]  # (1631, 4096, 12)
        assert tracings.ndim == 3 and tracings.shape[
            2] == 12, f"Shape inattesa: {tracings.shape}"
        assert len(
            df) == tracings.shape[0], f"CSV righe={len(df)} != H5 exams={tracings.shape[0]}"

        # 3) scegli un indice con death=False e uno con death=True
        idx_false, idx_true = pick_indices(df)

        # 4) preprocessa e plott a 5 secondi
        x_false = preprocess_trace(tracings[idx_false])  # (12, L500)
        mF = df.loc[idx_false]
        title_false = f"SaMi-Trop — death=False | exam_id={mF.get('exam_id', idx_false)} | age={mF.get('age', 'NA')} | normal_ecg={mF.get('normal_ecg','NA')}"
        plot_12leads(x_false, title_false, os.path.join(
            OUT_DIR, "samitrop_death_false_12leads.png"))
        plot_leadII(x_false, title_false, os.path.join(
            OUT_DIR, "samitrop_death_false_leadII.png"))

        x_true = preprocess_trace(tracings[idx_true])
        mT = df.loc[idx_true]
        title_true = f"SaMi-Trop — death=True | exam_id={mT.get('exam_id', idx_true)} | age={mT.get('age','NA')} | timey={mT.get('timey','NA')}"
        plot_12leads(x_true, title_true, os.path.join(
            OUT_DIR, "samitrop_death_true_12leads.png"))
        plot_leadII(x_true, title_true, os.path.join(
            OUT_DIR, "samitrop_death_true_leadII.png"))

    print("Figure salvate in:", OUT_DIR)


if __name__ == "__main__":
    main()

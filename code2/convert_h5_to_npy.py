import os
import h5py
import numpy as np
import pandas as pd
from utils import apply_filter, scaling  # preprocessing helpers


RAW_H5_DIR = r"F:\records\records_h5"
# nuovo: cartella per ECG preprocessati .npy
RAW_NPY_DIR = r"F:\records\records_npy_raw"
os.makedirs(RAW_NPY_DIR, exist_ok=True)

CSV_IN = [
    r"reproducibility\sph\sph_train.csv",
    r"reproducibility\sph\sph_val.csv",
    r"reproducibility\sph\sph_test.csv",
]
CSV_OUT = [
    r"reproducibility\sph\sph_train_abs.csv",
    r"reproducibility\sph\sph_val_abs.csv",
    r"reproducibility\sph\sph_test_abs.csv",
]


def read_ecg_from_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        # prova a trovare un dataset 2D con 12 canali
        def all_dsets(g):
            for k, v in g.items():
                if isinstance(v, h5py.Dataset):
                    yield v
                elif isinstance(v, h5py.Group):
                    yield from all_dsets(v)
        candidate = None
        for d in all_dsets(f):
            if d.ndim == 2 and 12 in d.shape:
                candidate = d[()]
                break
        if candidate is None:
            raise RuntimeError(f"Nessun dataset 12xN trovato in {h5_path}")
        # assicurati che la forma sia (12, N)
        arr = candidate if candidate.shape[0] == 12 else candidate.T
        return np.asarray(arr, dtype=np.float32)


def preprocess_12lead(ecg_12xN, original_fs=500, target_fs=500, band=[0.05, 47]):
    # (se ti servisse il resampling, è già importato come resample)
    ecg_12xN = apply_filter(ecg_12xN, band, fs=500)  # <- firma corretta (utils.apply_filter) 
    ecg_12xN = scaling(ecg_12xN)
    if ecg_12xN.shape[1] >= 2500:
        ecg_12xN = ecg_12xN[:, :2500]
    return ecg_12xN.astype(np.float32)


def h5_to_npy_if_missing(stem):
    # rimuovi qualunque estensione (.h5, .npy, ecc.) prima di comporre i path
    base = os.path.splitext(os.path.basename(stem))[0]
    h5_path = os.path.join(RAW_H5_DIR, base)
    npy_path = os.path.join(RAW_NPY_DIR, base + ".npy")

    if os.path.isfile(npy_path):
        return npy_path

    with h5py.File(h5_path, "r") as f:
        # trova un dataset 12xN
        def all_dsets(g):
            for k, v in g.items():
                if isinstance(v, h5py.Dataset):
                    yield v
                elif isinstance(v, h5py.Group):
                    yield from all_dsets(v)
        arr = None
        for d in all_dsets(f):
            if d.ndim == 2 and 12 in d.shape:
                arr = d[()]
                break
        if arr is None:
            raise RuntimeError(f"Nessun dataset 12xN trovato in {h5_path}")
        ecg = arr if arr.shape[0] == 12 else arr.T

    ecg = preprocess_12lead(ecg, original_fs=500, target_fs=500)
    np.save(npy_path, ecg)
    return npy_path


def rewrite_csv(csv_in, csv_out):
    df = pd.read_csv(csv_in, dtype={'filename': str})
    if 'filename' not in df.columns:
        raise ValueError(f"{csv_in} non ha colonna 'filename'")

    new_paths = []
    for fn in df['filename']:
        # accetta sia “A00001”, sia “A00001.h5”, sia path completi
        new_paths.append(h5_to_npy_if_missing(fn))
    df['filename'] = new_paths
    df.to_csv(csv_out, index=False)
    print(f"✔ CSV scritto: {csv_out}")

if __name__ == "__main__":
    for ci, co in zip(CSV_IN, CSV_OUT):
        rewrite_csv(ci, co)

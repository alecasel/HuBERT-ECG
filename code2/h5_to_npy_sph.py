# h5_to_npy_sph.py
from utils import ecg_preprocessing  # scaling + band-pass + resample to 500 Hz
import os
import sys
import glob
import warnings
import numpy as np
import h5py
from pathlib import Path

# importa il pre-processing del repo
# esegui questo script dalla cartella "code/"
sys.path.append(str(Path(__file__).resolve().parent))

# === CONFIG ===
# cartella con i .h5
H5_DIR = r"F:\records\records_h5"
# dove salvare i .npy
OUT_DIR = r"F:\records\records_npy"
DEFAULT_FS = 500  # se nel file .h5 non c'è fs, useremo questo

# nomi chiave comuni da provare dentro all'h5
LIKELY_DATA_KEYS = ["ecg", "ECG", "signal", "signals", "data", "val", "record"]


def _find_ecg_dataset(h5: h5py.File):
    # 1) prova alcune chiavi comuni alla radice
    for k in LIKELY_DATA_KEYS:
        if k in h5 and isinstance(h5[k], h5py.Dataset):
            return h5[k][()]
    # 2) altrimenti cerca la prima dataset plausibile (12×L oppure L×12)

    def _is_12lead(shape):
        return (len(shape) == 2) and (12 in shape) and (max(shape) > 100)
    for _, obj in h5.items():
        if isinstance(obj, h5py.Dataset) and _is_12lead(obj.shape):
            return obj[()]
        if isinstance(obj, h5py.Group):
            for subk, subobj in obj.items():
                if isinstance(subobj, h5py.Dataset) and _is_12lead(subobj.shape):
                    return subobj[()]
    raise KeyError("ECG dataset non trovato dentro l'h5")


def _get_fs(h5: h5py.File, default_fs=500):
    # prova negli attrs globali
    for key in ["fs", "FS", "sampling_rate", "SamplingRate", "sample_rate", "SampleRate"]:
        if key in h5.attrs:
            try:
                return int(h5.attrs[key])
            except Exception:
                pass
    # prova negli attrs dei dataset plausibili
    for k in h5.keys():
        obj = h5[k]
        if isinstance(obj, h5py.Dataset):
            for key in ["fs", "FS", "sampling_rate", "SamplingRate", "sample_rate", "SampleRate"]:
                if key in obj.attrs:
                    try:
                        return int(obj.attrs[key])
                    except Exception:
                        pass
    warnings.warn(
        f"Frequenza di campionamento non trovata: uso default {default_fs} Hz")
    return default_fs


def convert_one(h5_path: str, out_dir: str):
    base = os.path.splitext(os.path.basename(h5_path))[0]
    out_path = os.path.join(out_dir, base + ".npy")
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        arr = _find_ecg_dataset(h5)  # numpy array
        fs = _get_fs(h5, DEFAULT_FS)

    # porta la forma a (12, L)
    if arr.ndim != 2:
        raise ValueError(
            f"{h5_path}: atteso array 2D, trovato shape {arr.shape}")
    if arr.shape[0] == 12:
        ecg = arr
    elif arr.shape[1] == 12:
        ecg = arr.T
    else:
        raise ValueError(
            f"{h5_path}: nessuna dimensione pari a 12, shape={arr.shape}")

    ecg = np.asarray(ecg, dtype=np.float32)

    # pre-processing del repo: resample->500Hz, band-pass, scaling [-1,1]
    # Nota: ecg_preprocessing richiede (12, L) e original_frequency
    # repo ha 500Hz come riferimento
    ecg_pp = ecg_preprocessing(ecg, original_frequency=int(fs))

    # salva
    np.save(out_path, ecg_pp.astype(np.float32))
    return out_path


def main():
    h5_files = sorted(glob.glob(os.path.join(H5_DIR, "*.h5")))
    if not h5_files:
        print(f"Nessun .h5 trovato in {H5_DIR}")
        return
    ok, fail = 0, 0
    for f in h5_files:
        try:
            out = convert_one(f, OUT_DIR)
            print(f"[OK] {f} -> {out}")
            ok += 1
        except Exception as e:
            print(f"[ERR] {f}: {e}")
            fail += 1
    print(f"\nFatto. Convertiti: {ok}, errori: {fail}. Output in: {OUT_DIR}")


if __name__ == "__main__":
    main()

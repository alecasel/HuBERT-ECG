import os, h5py, numpy as np
from utils import apply_filter, scaling  # dal repo
from pathlib import Path

H5_DIR  = Path(r"F:\records\records_h5")        # originali .h5
RAW_DIR = Path(r"F:\records\records_npy_raw")   # RAW 12xL .npy da riparare (in-place)

def find_12xN_dataset(f: h5py.File):
    def all_dsets(g):
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                yield v
            elif isinstance(v, h5py.Group):
                yield from all_dsets(v)
    for d in all_dsets(f):
        if d.ndim == 2 and 12 in d.shape and max(d.shape) >= 2500:
            arr = d[()]
            return arr if arr.shape[0] == 12 else arr.T
    return None

def preprocess_12lead(ecg_12xN, fs=500):
    # filtra + scala come nel repo, poi taglia a 5s (2500 @500 Hz)
    ecg_12xN = apply_filter(ecg_12xN, [0.05, 47], fs=fs)
    ecg_12xN = scaling(ecg_12xN)
    if ecg_12xN.shape[1] >= 2500:
        ecg_12xN = ecg_12xN[:, :2500]
    return ecg_12xN.astype(np.float32)

def is_bad_raw(p: Path) -> bool:
    try:
        a = np.load(p, allow_pickle=True)
        if a.size == 0: return True
        if a.ndim != 2: return True
        if a.shape[0] != 12 and a.shape[1] == 12:
            a = a.T
        if a.shape[0] != 12: return True
        if a.shape[1] < 2500: return True
        return False
    except Exception:
        return True

def repair_one(stem: str):
    h5 = H5_DIR / f"{stem}.h5"
    npy = RAW_DIR / f"{stem}.h5.npy"
    if not h5.is_file():
        print(f"[MISS] H5 mancante: {h5}")
        return False
    try:
        with h5py.File(h5, "r") as f:
            ecg = find_12xN_dataset(f)
        if ecg is None:
            print(f"[FAIL] Nessun dataset 12xN in {h5}")
            return False
        ecg = preprocess_12lead(ecg, fs=500)
        tmp = RAW_DIR / f".{stem}.tmp.npy"
        np.save(tmp, ecg)
        os.replace(tmp, npy)
        print(f"[OK] Riparato {npy.name}  -> shape {ecg.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] {stem}: {e}")
        return False

def main():
    bad = []
    for p in RAW_DIR.glob("*.npy"):
        if is_bad_raw(p):
            bad.append(p.stem.split(".h5")[0])  # estrae "A12345" da "A12345.h5"
    print(f"Trovati {len(bad)} RAW problematici.")
    repaired = 0
    for stem in bad:
        repaired += repair_one(stem)
    print(f"Riparati: {repaired}/{len(bad)}")

if __name__ == "__main__":
    main()

import sys
import os

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

from code2.utils import ecg_preprocessing

import h5py
import numpy as np

# === PERCORSI (adatta solo se servono) ===
H5_PATH = r"F:\SaMi-Trop dataset\exams.hdf5"
OUT_DIR = r"F:\SaMi-Trop dataset\records_npy"
CSV_OUT = os.path.join(OUT_DIR, "sph_exams_hubert.csv")

ORIG_FS = 400  # Hz del file "exams.hdf5"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with h5py.File(H5_PATH, "r") as h5:
        if "tracings" not in h5:
            raise KeyError(
                "Nel file non esiste un dataset chiamato 'tracings'.")
        tracings = h5["tracings"]  # (N, 4096, 12)
        assert tracings.ndim == 3 and tracings.shape[
            2] == 12, f"Shape inattesa: {tracings.shape}"
        N = tracings.shape[0]

        written = []
        for i in range(N):
            x = tracings[i]              # (4096, 12)
            x = x.astype(np.float32).T   # -> (12, 4096)

            # Preprocessing del repo: resample a 500 Hz + band-pass + scaling [-1,1]
            x_pp = ecg_preprocessing(
                x, original_frequency=ORIG_FS)  # (12, L_500Hz)

            # Salva: es. SPH_00001.npy, …
            fname = f"SPH_{i+1:05d}.npy"
            np.save(os.path.join(OUT_DIR, fname), x_pp.astype(np.float32))
            written.append(fname)

            if (i+1) % 100 == 0:
                print(f"[{i+1}/{N}] {fname}")

    # CSV semplice con colonna 'filename' (compatibile con ECGDataset)
    # Il tuo loader legge (12, qualsiasi L) e poi usa 5 s/2500 campioni @500 Hz, quindi va bene.
    import pandas as pd
    pd.DataFrame({"filename": written}).to_csv(CSV_OUT, index=False)
    print(f"Fatto. Scritti {len(written)} file in {OUT_DIR}")
    print(f"CSV creato: {CSV_OUT}")


if __name__ == "__main__":
    main()

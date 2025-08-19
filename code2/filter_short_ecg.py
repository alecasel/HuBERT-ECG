# code/filter_short_ecg.py
import argparse
import os
import numpy as np
import pandas as pd


def resolve_path(root, fn):
    p = fn if os.path.isfile(fn) else os.path.join(root, fn)
    return p


def load_len_12xN(path):
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception as e:
        return None, f"load_error: {e}"
    if arr is None or arr.size == 0:
        return None, "empty_or_none"
    if arr.ndim != 2 or (12 not in arr.shape):
        return None, f"bad_shape:{getattr(arr,'shape',None)}"
    if arr.shape[0] == 12:
        ecg = arr
    elif arr.shape[1] == 12:
        ecg = arr.T
    else:
        return None, f"not_12_lead:{arr.shape}"
    return ecg, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV con colonna 'filename'")
    ap.add_argument("--ecg-dir", required=True,
                    help="Directory dove cercare i .npy se 'filename' non è assoluto")
    ap.add_argument("--min-samples", type=int, default=2500,
                    help="Soglia minima di campioni (5s=2500, 10s=5000)")
    ap.add_argument("--out-csv", required=True, help="CSV pulito in output")
    ap.add_argument("--log", required=True,
                    help="File di testo con gli scartati e motivazione")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype={'filename': str})
    if "filename" not in df.columns:
        raise ValueError(f"{args.csv} non contiene la colonna 'filename'")

    kept_rows = []
    discarded = []

    for i, row in df.iterrows():
        fn = row["filename"]
        p = resolve_path(args.ecg_dir, fn)
        if not os.path.isfile(p):
            discarded.append((fn, "missing_file"))
            continue
        ecg, err = load_len_12xN(p)
        if err:
            discarded.append((p, err))
            continue
        L = ecg.shape[1]
        if L < args.min_samples:
            discarded.append((p, f"too_short:{L}<{args.min_samples}"))
            continue
        kept_rows.append(row)

    df_out = pd.DataFrame(kept_rows)
    df_out.to_csv(args.out_csv, index=False)

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "w", encoding="utf-8") as f:
        f.write(f"# Soglia minima: {args.min_samples} campioni\n")
        f.write(
            f"# Totale input: {len(df)} | Tenuti: {len(df_out)} | Scartati: {len(discarded)}\n\n")
        for p, reason in discarded:
            f.write(f"{p}\t{reason}\n")

    print(f"✅ CSV pulito: {args.out_csv}")
    print(f"📝 Scartati: {args.log}")
    print(
        f"Riepilogo -> input: {len(df)} | tenuti: {len(df_out)} | scartati: {len(discarded)}")


if __name__ == "__main__":
    main()

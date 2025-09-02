
import argparse
import os
import re
import sys
import glob
import numpy as np
import pandas as pd


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate test.py --save_probs outputs into a single CSV with filenames and predictions.")
    ap.add_argument("--csv", required=True,
                    help="Path to the CSV used for testing (e.g., test_names_cls2.csv)")
    ap.add_argument("--probs_dir", required=True,
                    help="Directory where test.py saved the .npy files (e.g., probs/test_names_cls2)")
    ap.add_argument("--out_csv", required=True,
                    help="Where to write the aggregated predictions CSV")
    ap.add_argument(
        "--task", choices=["multi_label", "multi_class"], default="multi_label")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold for multi_label decisions")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # discover .npy files
    npy_files = sorted(glob.glob(os.path.join(
        args.probs_dir, "*.npy")), key=natural_sort_key)
    if not npy_files:
        sys.exit(
            f"Nessun .npy trovato in {args.probs_dir}. Hai passato --save_probs a test.py?")
    arrs = [np.load(p) for p in npy_files]
    probs = np.concatenate(arrs, axis=0)  # [N, C]
    if len(df) != len(probs):
        print(
            f"[WARN] len(CSV)={len(df)} != len(probs)={len(probs)}. Provo a troncare al minimo comune.", file=sys.stderr)
    n = min(len(df), len(probs))
    df = df.iloc[:n].reset_index(drop=True)
    probs = probs[:n]

    # Build output
    out = df[["filename"]].copy()
    if args.task == "multi_label":
        # assume 2 columns in CSV are normal_ecg and death if present
        lbl_cols = [c for c in df.columns if c.lower() in (
            "normal_ecg", "death")]
        C = probs.shape[1]
        # Ensure we have exactly 2 columns
        names = lbl_cols if len(lbl_cols) == C else [
            f"label_{i}" for i in range(C)]
        for i, name in enumerate(names):
            out[f"prob_{name}"] = probs[:, i]
            out[f"pred_{name}"] = (probs[:, i] >= args.threshold).astype(int)
        # include GT if present
        for name in ("normal_ecg", "death"):
            if name in df.columns:
                out[f"gt_{name}"] = df[name].astype(int)
    else:
        # multi_class
        out["pred_class"] = probs.argmax(axis=1)
        for i in range(probs.shape[1]):
            out[f"prob_class_{i}"] = probs[:, i]

    out.to_csv(args.out_csv, index=False)
    print(f"OK: scritto {args.out_csv} con {len(out)} righe.")


if __name__ == "__main__":
    main()

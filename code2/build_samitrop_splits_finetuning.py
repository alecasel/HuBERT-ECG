
import argparse
import os
import re
import sys
import pandas as pd
import numpy as np

try:
    from sklearn.model_selection import StratifiedShuffleSplit
    HAS_SK = True
except Exception:
    HAS_SK = False


def extract_exam_id_from_filename(s: str):
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else None


def coerce_binary(x):
    # Coerce to integer 0/1 in a tolerant way
    if pd.isna(x):
        return 0
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    try:
        f = float(x)
        return int(1 if f != 0 else 0)
    except Exception:
        s = str(x).strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return 1
        return 0


def load_exams_build_map(path_exams: str, raw_dir: str):
    ex = pd.read_csv(path_exams).reset_index().rename(
        columns={"index": "h5_index"})
    # Find exam_id column or infer
    exam_col = None
    for cand in ["exam_id", "id", "ExamID", "examid"]:
        if cand in ex.columns:
            exam_col = cand
            break
    if exam_col is None:
        if "filename" in ex.columns:
            ex["exam_id"] = ex["filename"].astype(
                str).apply(extract_exam_id_from_filename)
            exam_col = "exam_id"
        else:
            raise ValueError(
                "Impossibile trovare la colonna exam_id in exams.csv.")
    ex["sph_index"] = ex["h5_index"] + 1
    ex["sph_file"] = ex["sph_index"].apply(lambda k: f"SPH_{k:05d}.npy")
    ex["sph_path"] = ex["sph_file"].apply(
        lambda s: os.path.join(raw_dir, s) if raw_dir else s)
    exam_map = ex[[exam_col, "h5_index", "sph_file", "sph_path"]].rename(columns={
                                                                         exam_col: "exam_id"})
    if not exam_map["exam_id"].is_unique:
        raise ValueError("exam_id duplicati in exams.csv.")
    return exam_map


def normalize_split_df(df: pd.DataFrame, exam_map: pd.DataFrame, keep_abs: bool, raw_dir: str):
    if "filename" not in df.columns:
        raise ValueError("CSV di input: manca la colonna 'filename'.")
    # Build exam_id from filename if present, else use provided exam_id column
    if "exam_id" not in df.columns:
        df["exam_id"] = df["filename"].apply(
            extract_exam_id_from_filename).astype("Int64")
    # Join
    out = df.merge(exam_map, on="exam_id", how="left", validate="m:1")
    if out["h5_index"].isna().any():
        missing = out[out["h5_index"].isna()][["filename", "exam_id"]].head()
        raise ValueError(
            f"{out['h5_index'].isna().sum()} righe non mappate; esempi:\n{missing}")
    # Labels: require columns normal_ecg and death
    if "normal_ecg" not in out.columns or "death" not in out.columns:
        raise ValueError("Nei CSV sorgente mancano 'normal_ecg' o 'death'.")
    out["normal_ecg"] = out["normal_ecg"].apply(coerce_binary).astype(int)
    out["death"] = out["death"].apply(coerce_binary).astype(int)
    # Compose final
    out_filename = out["sph_path" if keep_abs else "sph_file"]
    final = pd.DataFrame({
        "filename": out_filename,
        "normal_ecg": out["normal_ecg"].astype(int),
        "death": out["death"].astype(int),
    })
    # Optional check: files exist
    if raw_dir:
        does_exist = final["filename"].apply(lambda f: os.path.isabs(f) and os.path.exists(
            f) or os.path.exists(os.path.join(raw_dir, os.path.basename(f))))
        miss = (~does_exist).sum()
        if miss:
            sample = final.loc[~does_exist, "filename"].head().tolist()
            print(
                f"[ATTENZIONE] {miss} file non trovati nel raw_dir. Esempi: {sample}", file=sys.stderr)
    return final


def split_val_from_train(train_df: pd.DataFrame, val_frac: float, stratify_on: str = "death", seed: int = 42):
    if val_frac <= 0 or val_frac >= 1:
        raise ValueError("val_frac deve essere nel range (0,1).")
    y = train_df[stratify_on] if stratify_on in train_df.columns else None
    if y is not None and HAS_SK and len(np.unique(y)) > 1:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac, random_state=seed)
        idx_train, idx_val = next(sss.split(train_df, y))
        tr = train_df.iloc[idx_train].reset_index(drop=True)
        va = train_df.iloc[idx_val].reset_index(drop=True)
    else:
        va = train_df.sample(frac=val_frac, random_state=seed)
        tr = train_df.drop(index=va.index).reset_index(drop=True)
        va = va.reset_index(drop=True)
    return tr, va


def main():
    ap = argparse.ArgumentParser(
        description="Costruisce train/val/test CSV con 2 label (normal_ecg, death) mappando i filename su SPH_XXXXX.npy.")
    ap.add_argument("--exams", required=True, help="Percorso a exams.csv")
    ap.add_argument("--raw-dir", required=True,
                    help="Cartella con i RAW SPH_XXXXX.npy (records_npy_raw)")
    ap.add_argument("--train-src", required=True,
                    help="CSV sorgente train (può avere filename samitrop_*.npy)")
    ap.add_argument("--test-src",  required=True, help="CSV sorgente test")
    ap.add_argument("--val-src",   default=None,
                    help="CSV sorgente val (se assente, viene creato dallo split del train)")
    ap.add_argument("--val-frac",  type=float, default=0.10,
                    help="Quota di validation quando deriva dal train")
    ap.add_argument("--out-dir",   required=True, help="Cartella di output")
    ap.add_argument("--prefix",    default="names_cls2",
                    help="Suffisso base per i CSV di output (es: names_cls2)")
    ap.add_argument("--abs-paths", action="store_true",
                    help="Scrive filename come percorsi assoluti invece che nomi SPH_*.npy")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    exam_map = load_exams_build_map(args.exams, args.raw_dir)

    def read_csv(path):
        return pd.read_csv(path)

    train_src = read_csv(args.train_src)
    test_src = read_csv(args.test_src)
    val_src = read_csv(args.val_src) if args.val_src else None

    # Normalize/match -> SPH and reduce to 2 labels
    train_full = normalize_split_df(
        train_src, exam_map, keep_abs=args.abs_paths, raw_dir=args.raw_dir)
    test_full = normalize_split_df(
        test_src,  exam_map, keep_abs=args.abs_paths, raw_dir=args.raw_dir)

    if val_src is not None:
        val_full = normalize_split_df(
            val_src, exam_map, keep_abs=args.abs_paths, raw_dir=args.raw_dir)
    else:
        train_full, val_full = split_val_from_train(
            train_full, args.val_frac, stratify_on="death")

    # Save
    def save(df, name):
        path = os.path.join(args.out_dir, name)
        df.to_csv(path, index=False)
        return path

    out_train = save(train_full, f"train_{args.prefix}.csv")
    out_val = save(val_full,   f"val_{args.prefix}.csv")
    out_test = save(test_full,  f"test_{args.prefix}.csv")

    # Quick summaries
    def dist(frame, col):
        return dict(frame[col].value_counts(dropna=False).sort_index())

    print("== FATTO ==")
    print("Train:", out_train, "rows:", len(train_full), "death dist:", dist(
        train_full, "death"), "normal_ecg dist:", dist(train_full, "normal_ecg"))
    print("Val:  ", out_val,   "rows:", len(val_full),   "death dist:", dist(
        val_full,   "death"), "normal_ecg dist:", dist(val_full,   "normal_ecg"))
    print("Test: ", out_test,  "rows:", len(test_full),  "death dist:", dist(
        test_full,  "death"), "normal_ecg dist:", dist(test_full,  "normal_ecg"))


if __name__ == "__main__":
    main()

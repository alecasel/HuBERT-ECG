from pathlib import Path
import pandas as pd
import re
import sys
import os

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))


# === CONFIG ===
EXAMS_CSV = r"F:\SaMi-Trop dataset\exams.csv"
SPH_DIR = r"F:\SaMi-Trop dataset\records_npy_raw"

# I tre CSV da mappare
INPUT_CSVS = [
    r"C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG\reproducibility\samitrop\train_my.csv",
    r"C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG\reproducibility\samitrop\val_my.csv",
    r"C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG\reproducibility\samitrop\test_my.csv",
]

# Scegli se sostituire 'filename' con solo il nome SPH o con il percorso completo
REPLACE_WITH_FULL_PATH = False  # True => path completo; False => solo nome file

# === LOAD EXAMS & BUILD MAP ===
exams = pd.read_csv(EXAMS_CSV).reset_index().rename(
    columns={"index": "h5_index"})

# Infer exam_id column if missing
EXAM_ID_COL = None
for cand in ["exam_id", "id", "examid", "ExamID"]:
    if cand in exams.columns:
        EXAM_ID_COL = cand
        break

if EXAM_ID_COL is None:
    # come fallback, prova a derivarlo da una colonna 'filename' stile "samitrop_XXXXXX.npy" se presente
    if "filename" in exams.columns:
        exams["exam_id"] = exams["filename"].astype(
            str).str.extract(r"(\d+)").astype(int)
        EXAM_ID_COL = "exam_id"
    else:
        raise ValueError(
            "Non trovo la colonna exam_id in exams.csv. Imposta EXAM_ID_COL manualmente.")

# costruisci nome SPH
exams["sph_index"] = exams["h5_index"] + 1  # da 1 a 1631
exams["sph_file"] = exams["sph_index"].apply(lambda k: f"SPH_{k:05d}.npy")
exams["sph_path"] = exams["sph_file"].apply(lambda s: str(Path(SPH_DIR) / s))

exam_map = exams[[EXAM_ID_COL, "h5_index", "sph_file", "sph_path"]].rename(
    columns={EXAM_ID_COL: "exam_id"})

# controlli base
assert exam_map["exam_id"].is_unique, "exam_id duplicati in exams.csv: controlla il file!"


def extract_exam_id_from_filename(s):
    m = re.search(r"(\d+)", str(s))
    if not m:
        return None
    return int(m.group(1))


def process_one_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    if "filename" not in df.columns:
        raise ValueError(f"{csv_path}: non trovo la colonna 'filename'")

    # exam_id dai filename 'samitrop_XXXXXX.npy'
    df["exam_id"] = df["filename"].apply(
        extract_exam_id_from_filename).astype("Int64")

    # join con la mappa
    out = df.merge(exam_map, on="exam_id", how="left", validate="m:1")

    # sanity checks
    missing = out["h5_index"].isna().sum()
    if missing:
        missing_rows = out[out["h5_index"].isna()][["filename", "exam_id"]]
        raise ValueError(
            f"{csv_path}: {missing} righe non mappate.\n{missing_rows.head()}")

    # verifica esistenza file SPH sul disco (opzionale)
    try:
        from pathlib import Path
        exists_mask = out["sph_path"].apply(lambda p: Path(p).exists())
        if (~exists_mask).any():
            nmiss = (~exists_mask).sum()
            print(
                f"ATTENZIONE: {nmiss} file SPH non trovati su disco per {csv_path} (controlla SPH_DIR).")
    except Exception:
        pass

    # --- Versione 1: arricchita (aggiunge colonne di mappatura per debug)
    enriched_path = str(Path(csv_path).with_name(
        Path(csv_path).stem + "_mapped_enriched.csv"))
    out.to_csv(enriched_path, index=False)

    # --- Versione 2: stesso schema dell’input, ma filename -> SPH
    out2 = df.copy()
    out2["filename"] = out["sph_path" if REPLACE_WITH_FULL_PATH else "sph_file"]
    clean_path = str(Path(csv_path).with_name(
        Path(csv_path).stem + "_mapped.csv"))
    out2.to_csv(clean_path, index=False)

    print(f"OK: {csv_path}\n  → {enriched_path}\n  → {clean_path}")


for csv in INPUT_CSVS:
    process_one_csv(csv)

print("Fatto ✅")

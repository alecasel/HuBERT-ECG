# CONFIG
$PY   = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\.venv\Scripts\python.exe"
$PROJ = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG"

$CSV_TRAIN_NAMES = Join-Path $PROJ "reproducibility\sph\sph_train_names.csv"
$CSV_VAL_NAMES   = Join-Path $PROJ "reproducibility\sph\sph_val_names.csv"

$CSV_TRAIN_SMALL = Join-Path $PROJ "reproducibility\sph\sph_train_small.csv"
$CSV_VAL_SMALL   = Join-Path $PROJ "reproducibility\sph\sph_val_small.csv"

$N_TRAIN = 5000
$N_VAL   = 1000

# CREA SOTTOINSIEME (random con seed fisso)
@"
import pandas as pd, numpy as np
rng = np.random.RandomState(42)
df_tr = pd.read_csv(r"$CSV_TRAIN_NAMES", dtype={'filename':str})
df_va = pd.read_csv(r"$CSV_VAL_NAMES",   dtype={'filename':str})

n_tr = min($N_TRAIN, len(df_tr))
n_va = min($N_VAL,   len(df_va))

df_tr_small = df_tr.sample(n=n_tr, random_state=42).reset_index(drop=True)
df_va_small = df_va.sample(n=n_va, random_state=42).reset_index(drop=True)

df_tr_small.to_csv(r"$CSV_TRAIN_SMALL", index=False)
df_va_small.to_csv(r"$CSV_VAL_SMALL",   index=False)

print("train_small:", len(df_tr_small), "val_small:", len(df_va_small))
"@ | & $PY -

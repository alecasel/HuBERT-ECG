# ============ CONFIG ============
$PY   = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\.venv\Scripts\python.exe"
$PROJ = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG"

# CSV con path assoluti agli ECG preprocessati (.npy 12×N) — ADATTA questi nomi se non usi il suffisso _abs
$CSV_TRAIN = Join-Path $PROJ "reproducibility\samitrop\train_my_mapped.csv"
$CSV_VAL   = Join-Path $PROJ "reproducibility\samitrop\val_my_mapped.csv"
$CSV_TEST  = Join-Path $PROJ "reproducibility\samitrop\test_my_mapped.csv"

# Cartelle dati
$NPY_RAW   = "F:\SaMi-Trop dataset\records_npy_raw"  # input: ECG 12×N preprocessati
$FEAT_DIR  = "F:\SaMi-Trop dataset\records_npy"      # output: descrittori (93×29) per ECG

# ============ RUN ============
Set-Location $PROJ
New-Item -ItemType Directory -Path $FEAT_DIR -Force | Out-Null

# Train
& $PY code2/dumping.py 1 `
    $CSV_TRAIN `
    $NPY_RAW `
    $FEAT_DIR `
    0.0 1.0 --samp_rate 100 --input_fs 400

# Val
& $PY code2/dumping.py 1 `
    $CSV_VAL `
    $NPY_RAW `
    $FEAT_DIR `
    0.0 1.0 --samp_rate 100 --input_fs 400

# Test
& $PY code2/dumping.py 1 `
    $CSV_TEST `
    $NPY_RAW `
    $FEAT_DIR `
    0.0 1.0 --samp_rate 100 --input_fs 400

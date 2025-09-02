# ============ CONFIG ============
$PY   = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\.venv\Scripts\python.exe"
$PROJ = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG"

# CSV assoluti (input) — ADATTA se i nomi differiscono
$CSV_TRAIN_ABS = Join-Path $PROJ "reproducibility\samitrop\train_my_mapped.csv"
$CSV_VAL_ABS   = Join-Path $PROJ "reproducibility\samitrop\val_my_mapped.csv"
$CSV_TEST_ABS  = Join-Path $PROJ "reproducibility\samitrop\test_my_mapped.csv"

# CSV “names” (output) con SOLO il nome file (es. A00001.npy)
$CSV_TRAIN_NAMES = Join-Path $PROJ "reproducibility\samitrop\train_names.csv"
$CSV_VAL_NAMES   = Join-Path $PROJ "reproducibility\samitrop\val_names.csv"
$CSV_TEST_NAMES  = Join-Path $PROJ "reproducibility\samitrop\test_names.csv"

# Directory dei descrittori (output dello Step 1)
$FEAT_DIR = "F:\SaMi-Trop dataset\records_npy"

# Clustering params
$N_CLUSTERS = 100
$BATCH_SIZE = 16
$TRAIN_ITER = 1   # iteration 1 => “morphology” nel nome file del modello

# ============ RUN ============
Set-Location $PROJ

# 1) Genera CSV “names” (solo nome file)
@"
import pandas as pd, os
def mk(src,dst):
    df = pd.read_csv(src, dtype={'filename':str})
    df['filename'] = df['filename'].apply(lambda p: os.path.basename(p))
    df.to_csv(dst, index=False)
mk(r"$CSV_TRAIN_ABS", r"$CSV_TRAIN_NAMES")
mk(r"$CSV_VAL_ABS",   r"$CSV_VAL_NAMES")
mk(r"$CSV_TEST_ABS",  r"$CSV_TEST_NAMES")
"@ | & $PY -

# 2) Disattiva WandB (gli script lo chiamano di default)
$env:WANDB_MODE = "disabled"

# 3) Clustering (usa i descrittori in $FEAT_DIR) — salva un .pkl nel $PROJ
& $PY code2/cluster.py `
    $CSV_TRAIN_NAMES `
    $FEAT_DIR `
    --cluster `
    --n_clusters_start $N_CLUSTERS `
    --n_clusters_end   $N_CLUSTERS `
    --step 1 `
    $TRAIN_ITER `
    $BATCH_SIZE

# 4) Crea kmeans.txt con l’ultimo modello salvato (serve al pretraining)
$KMODEL = Get-ChildItem -Path $PROJ -Filter "k_means_*_morphology_*.pkl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $KMODEL) { throw "Nessun modello KMeans trovato." }

$KMEANS_TXT = Join-Path $PROJ "kmeans.txt"
$KMODEL.FullName | Out-File -FilePath $KMEANS_TXT -Encoding ascii
Write-Host "kmeans.txt ->" (Get-Content $KMEANS_TXT)

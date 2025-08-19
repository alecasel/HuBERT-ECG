# CONFIG
$PY   = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\.venv\Scripts\python.exe"
$PROJ = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG"

$CSV_TRAIN_SMALL = Join-Path $PROJ "reproducibility\sph\sph_train_small.csv"
$CSV_VAL_SMALL   = Join-Path $PROJ "reproducibility\sph\sph_val_small.csv"

$KMEANS_TXT = Join-Path $PROJ "kmeans.txt"   # va benissimo quello addestrato su tutto

# CARTELLE DATI
$FEAT_DIR = "F:\records\records_npy"       # <-- features 93×29 (CORRETTO)
# i RAW sono già presi da pretrain.py (ecg_dir_path = F:\records\records_npy_raw)

# PARAMETRI “CPU”
$VAL_INTERVAL   = 500
$MASK_TIME_PROB = 0.065
$BATCH_SIZE     = 2
$MODEL_SIZE     = "small"
$ALPHA          = 0.5
$VOCAB_SIZE     = 100
$TRAINING_STEPS = 1000    # es. 1k step; aumenta a piacere

$env:WANDB_MODE = "disabled"
Set-Location $PROJ

& $PY code\pretrain.py 1 `
  $CSV_TRAIN_SMALL `
  $CSV_VAL_SMALL `
  $VAL_INTERVAL `
  $MASK_TIME_PROB `
  $BATCH_SIZE `
  $MODEL_SIZE `
  $ALPHA `
  $KMEANS_TXT `
  $FEAT_DIR `
  $FEAT_DIR `
  $VOCAB_SIZE `
  --training_steps $TRAINING_STEPS

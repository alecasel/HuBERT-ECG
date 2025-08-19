# Python del venv
$py = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\.venv\Scripts\python.exe"

# CSV (stessa struttura per train e val)
$CSV_TRAIN = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG\reproducibility\sph\sph_train_small.csv"
$CSV_VAL   = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG\reproducibility\sph\sph_val_small.csv"

# per evitare WandB
$env:WANDB_MODE = "disabled"

# POSIZIONALI (in quest’ordine!) + OPZIONI
& $py code\finetune.py 1 `
  $CSV_TRAIN `
  $CSV_VAL `
  44 `
  6 `
  8 `
  f1_score `
  --epochs 10 `
  --task multi_label `
  --label_start_index 4 `
  --downsampling_factor 5 `
  --load_path "F:\models\hubert_ecg_small_hf_init.pt"

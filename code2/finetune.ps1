# Python del venv
$py = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\.venv\Scripts\python.exe"

# CSV (stessa struttura per train e val)
$CSV_TRAIN = "reproducibility\samitrop\train_names_cls2.csv"
$CSV_VAL   = "reproducibility\samitrop\val_names_cls2.csv"

# per evitare WandB
$env:WANDB_MODE = "disabled"

# POSIZIONALI (in quest’ordine!) + OPZIONI
& $py code2\finetune.py 1 `
  $CSV_TRAIN `
  $CSV_VAL `
  2 `
  6 `
  8 `
  f1_score `
  --epochs 100 `
  --task multi_label `
  --label_start_index 1 `
  --downsampling_factor 4 `
  --load_path "F:\models\hubert_ecg_small_hf_init.pt" `
  --ecg_dir_path "F:\SaMi-Trop dataset\records_npy_raw"

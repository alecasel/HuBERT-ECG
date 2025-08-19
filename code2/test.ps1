$py   = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\.venv\Scripts\python.exe"
$PROJ = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG"
$CSV_TEST_SMALL = Join-Path $PROJ "reproducibility\sph\sph_test_small.csv"
$csv = $CSV_TEST_SMALL
$ecg = "F:\records\records_npy_raw"
$ckp = "F:\models\checkpoints\hubert_1_iteration_5000_finetuned_1j3zk6ia.pt"
$env:WANDB_MODE = "disabled"

& $py code\test.py $csv $ecg 8 $ckp `
  --label_start_index 4 `
  --downsampling_factor 5 `
  --task multi_label `
  --save_id sph_test_eval `
  --save_probs `

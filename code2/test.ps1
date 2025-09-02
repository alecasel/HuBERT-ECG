$py   = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\.venv\Scripts\python.exe"
$PROJ = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG"
$CSV_TEST_SMALL = Join-Path $PROJ "reproducibility\samitrop\test_names_cls2.csv"
$csv = $CSV_TEST_SMALL
$ecg = "F:\SaMi-Trop dataset\records_npy_raw"
$ckp = "F:\models\checkpoints_samitrop\hubert_1_iteration_1320_finetuned_9jsj3kgs.pt"
$env:WANDB_MODE = "disabled"

& $py code2\test.py $csv $ecg 8 $ckp `
  --label_start_index 1 `
  --downsampling_factor 4 `
  --task multi_label `
  --save_id sph_test_eval `
  --save_probs `

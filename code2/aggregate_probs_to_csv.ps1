$py = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\.venv\Scripts\python.exe"
$PROJ = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG"

$CSV = Join-Path $PROJ "reproducibility\samitrop\test_names_cls2.csv"
$PROBS_DIR = Join-Path $PROJ "probs\test_names_cls2"
$OUT = Join-Path $PROJ "reproducibility\samitrop\test_predictions_cls2.csv"

& $py "code2\aggregate_probs_to_csv.py" `
  --csv $CSV `
  --probs_dir $PROBS_DIR `
  --out_csv $OUT `
  --task multi_label `
  --threshold 0.5

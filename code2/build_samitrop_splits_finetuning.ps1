$py = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\.venv\Scripts\python.exe"

# file locali
$EXAMS = "F:\SaMi-Trop dataset\exams.csv"
$RAW   = "F:\SaMi-Trop dataset\records_npy_raw"

# sorgenti: prendi i tuoi split "di partenza"
$TRAIN_SRC = "reproducibility\samitrop\train_my_mapped.csv"   # o samitrop_train0.csv
$TEST_SRC  = "reproducibility\samitrop\test_my_mapped.csv"    # o samitrop_test0.csv
# se hai già un val sorgente, mettilo qui; altrimenti lascia vuoto per derivarlo dal train
$VAL_SRC   = "reproducibility\samitrop\val_my_mapped.csv" 

$OUT_DIR   = "reproducibility\samitrop"

# (Oppure, se hai anche un VAL sorgente)
& $py "code2\build_samitrop_splits_finetuning.py" `
  --exams   $EXAMS `
  --raw-dir $RAW `
  --train-src $TRAIN_SRC `
  --test-src  $TEST_SRC `
  --val-src   $VAL_SRC `
  --out-dir   $OUT_DIR `
  --prefix    "names_cls2"

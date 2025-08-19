# CONFIG
$py   = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\.venv\Scripts\python.exe"
$PROJ = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG"
$CSV_TEST_NAMES = Join-Path $PROJ "reproducibility\sph\sph_test_names.csv"
$LABELS_MASTER  = "reproducibility\sph\sph_test.csv"
$CSV_TEST_LABELED = Join-Path $PROJ "reproducibility\sph\sph_test_labeled.csv"
$CSV_TEST_SMALL = Join-Path $PROJ "reproducibility\sph\sph_test_small.csv"
$N = 500

@"
import pandas as pd
names = pd.read_csv(r"$CSV_TEST_NAMES", dtype={'filename':str})
lab   = pd.read_csv(r"$LABELS_MASTER",  dtype={'filename':str})
df = names[['filename']].merge(lab, on='filename', how='inner')
assert df.shape[1] >= 5, "Mi aspetto: filename, age, Patient_ID, sex + 44 label"
df.to_csv(r"$CSV_TEST_LABELED", index=False)
print("test_labeled righe:", len(df))
df = pd.read_csv(r"$CSV_TEST_LABELED")
n = min($N, len(df))
df.sample(n=n, random_state=42).to_csv(r"$CSV_TEST_SMALL", index=False)
print("test_small righe:", n)
"@ | & $py -

# Esegui il filtro sui .npy corti e scrivi un CSV pulito + log scartati

$py = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\.venv\Scripts\python.exe"
$repo = "C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG"

# === INPUT ===
# CSV originale usato nel test:
$csvIn  = "$repo\reproducibility\sph\sph_test_small.csv"

# Directory con i .npy (adatta se diverso: F:\records\records_npy_raw oppure F:\records\records_npy)
$ecgDir = "F:\records\records_npy_raw"

# === PARAMETRI ===
# 5s -> 2500 ; 10s -> 5000
$minSamples = 2500

# === OUTPUT ===
$outCsv = "$repo\reproducibility\sph\sph_test_small_len≥5s.csv"
$logTxt = "$repo\reproducibility\sph\sph_test_small_discarded.txt"

& $py "$repo\code\filter_short_ecg.py" `
  --csv "$csvIn" `
  --ecg-dir "$ecgDir" `
  --min-samples $minSamples `
  --out-csv "$outCsv" `
  --log "$logTxt"

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

# File path (adatta alle tue cartelle)
truth_csv = r"C:\Users\Alessandro\Desktop\workspace\ecg-online-sources\hubert-from-git\HuBERT-ECG\reproducibility\sph\sph_val_small.csv"
preds_csv = r"F:\models\probes\preds_val_linear_probe.csv"

# Carica
truth = pd.read_csv(truth_csv)
preds = pd.read_csv(preds_csv)

# Colonne di interesse
cols = ["CLBBB|LBBB", "IRBBB", "CRBBB|RBBB"]

# Subset ground truth (occhio: nel CSV del dataset le label partono dalla colonna 4)
truth_sub = truth[cols]

# Predizioni
preds_sub = preds[cols]

# Combina in unico target (1 se almeno una colonna è 1)
truth_rbbb = (truth_sub.max(axis=1) > 0).astype(int)
preds_rbbb = (preds_sub.max(axis=1) > 0).astype(int)

# Metriche
f1 = f1_score(truth_rbbb, preds_rbbb, zero_division=0)
prec = precision_score(truth_rbbb, preds_rbbb, zero_division=0)
rec = recall_score(truth_rbbb, preds_rbbb, zero_division=0)

print("Valutazione combinata (CLBBB|LBBB, IRBBB, CRBBB|RBBB)")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

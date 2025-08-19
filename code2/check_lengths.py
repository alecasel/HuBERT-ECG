import os
import numpy as np

NPY_DIR = r"F:\records\records_npy_raw"

lengths = []
for fn in os.listdir(NPY_DIR):
    if fn.endswith(".npy"):
        path = os.path.join(NPY_DIR, fn)
        try:
            arr = np.load(path, allow_pickle=True)
            if arr.ndim == 2:
                if arr.shape[0] == 12:
                    length = arr.shape[1]
                elif arr.shape[1] == 12:
                    length = arr.shape[0]
                else:
                    print(f"[SKIP] {fn} shape {arr.shape}")
                    continue
                lengths.append((fn, length))
            else:
                print(f"[BAD ] {fn} shape {arr.shape}")
        except Exception as e:
            print(f"[ERR ] {fn}: {e}")

print("\n--- Summary ---")
if lengths:
    lengths.sort(key=lambda x: x[1])
    for fn, length in lengths[:10]:
        print(f"{fn} -> {length} samples (shortest)")
    print("...")
    for fn, length in lengths[-10:]:
        print(f"{fn} -> {length} samples (longest)")
    lengths_only = [l for _, l in lengths]
    print(f"\nMin length = {min(lengths_only)}")
    print(f"Max length = {max(lengths_only)}")
    print(f"Num files  = {len(lengths_only)}")
else:
    print("No valid files found")

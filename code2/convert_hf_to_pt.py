# code/convert_hf_to_pt.py
import argparse, torch
from transformers import AutoConfig, AutoModel
from hubert_ecg import HuBERTECG as HuBERT, HuBERTECGConfig  # usa le classi locali

parser = argparse.ArgumentParser()
parser.add_argument("--hf_id", required=True)     # es. "Edoardo-BS/hubert-ecg-small"
parser.add_argument("--out", required=True)       # es. "F:\\models\\hubert_ecg_small_hf_init.pt"
args = parser.parse_args()

print(f"Downloading {args.hf_id} from HF...")
cfg_hf = AutoConfig.from_pretrained(args.hf_id, trust_remote_code=True)
sd_hf  = AutoModel.from_pretrained(args.hf_id, trust_remote_code=True).state_dict()

# Adatta la config HF alla config locale
cfg = HuBERTECGConfig(**cfg_hf.to_dict()) if not isinstance(cfg_hf, HuBERTECGConfig) else cfg_hf

# Costruisci il backbone locale e carica i pesi HF
model = HuBERT(cfg)
# Compatibilità eventuale su pos_conv_embed (alcune versioni rinominano i parametri)
new_sd = {}
for k, v in sd_hf.items():
    if k.endswith("parametrizations.weight.original0"):
        new_sd[k.replace("parametrizations.weight.original0","weight_g")] = v
    elif k.endswith("parametrizations.weight.original1"):
        new_sd[k.replace("parametrizations.weight.original1","weight_v")] = v
    else:
        new_sd[k] = v
missing, unexpected = model.load_state_dict(new_sd, strict=False)
print("Missing keys:", len(missing), "| Unexpected keys:", len(unexpected))

# Salva nel formato atteso dal finetune del progetto
checkpoint = {
    "global_step": 0,
    "best_val_loss": float("inf"),
    "model_config": model.config,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": {},
    "lr_scheduler_state_dict": {},
    "best_val_accuracy": 0.0,
}
torch.save(checkpoint, args.out)
print("Saved to", args.out)

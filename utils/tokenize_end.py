# utils/tokenize_end.py
# -*- coding: utf-8 -*-
"""
Build text embeddings for end.txt using selectable encoders (HF text models or KoCLIP).
Saves tensors under: pt_data/<model_key>_pt_end/<label_site>.pt

Saved file content:
{
  "text_emb": torch.Tensor(H),   # sentence embedding
  "label":    int                # class index from folder name prefix
}
"""
import os
import re
import time
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel, AutoProcessor

# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PT_DIR = ROOT_DIR / "pt_data"
PARSER_ERR_LOG = ROOT_DIR / "parser_error_log_end.txt"

# -----------------------------
# Encoder presets
#   key -> (backend, hf_id, description)
#   backend: "hf_text" (AutoTokenizer+AutoModel) | "koclip" (AutoProcessor+AutoModel(CLIP))
# -----------------------------
MODEL_PRESETS = {
    "koelectra": ("hf_text", "monologg/koelectra-base-v3-discriminator", "KoELECTRA base"),
    "kcbert":    ("hf_text", "beomi/KcBERT-base",                        "KcBERT base"),
    "koclip":    ("koclip",  "koclip/koclip-base-pt",                    "KoCLIP base (text)"),
}

# HTML tags to prioritize when extracting visible text
IMPORTANT_TAGS = ['title', 'h1', 'h2', 'h3', 'h4', 'td', 'p', 'a', 'b', 'span', 'button', 'li']

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================
# HTML utilities
# =============================
def clean_text(text: str) -> str:
    """Keep alphanumerics/whitespace/Korean chars; collapse spaces."""
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_text_from_html(html: str, max_char_len: int = 4096) -> str:
    """Parse HTML and extract a compact, readable string."""
    try:
        soup = BeautifulSoup(html, 'lxml')
    except Exception:
        soup = BeautifulSoup(html, 'html.parser')

    texts, total = [], 0
    for tag in IMPORTANT_TAGS:
        for elem in soup.find_all(tag):
            txt = elem.get_text(strip=True)
            if txt and len(txt) > 1:
                texts.append(txt)
                total += len(txt)
            if total > max_char_len:
                break
        if total > max_char_len:
            break

    combined = " ".join(texts) if texts else soup.get_text(separator=" ")
    return clean_text(combined)


# =============================
# Pooling
# =============================
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom  # (B, H)


def cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Use first token as sentence representation."""
    return last_hidden_state[:, 0, :]  # (B, H)


# =============================
# Main preprocessing
# =============================
def preprocess_all(model_key: str, pooling: str = "mean", max_len: int = 512):
    assert model_key in MODEL_PRESETS, f"Unknown model_key: {model_key}"
    backend, hf_id, _ = MODEL_PRESETS[model_key]

    # Output root
    save_root = PT_DIR / f"{model_key}_pt_end"
    os.makedirs(save_root, exist_ok=True)

    # Reset parser error log
    open(PARSER_ERR_LOG, "w").close()

    # Collect all (label_idx, end_path, save_path)
    all_items = []
    for label_folder in os.listdir(DATA_DIR):
        if "_" not in label_folder:
            continue
        label_idx = label_folder.split("_")[0]
        if not label_idx.isdigit():
            continue

        label_path = DATA_DIR / label_folder
        for site in os.listdir(label_path):
            site_path = label_path / site
            if not site_path.is_dir():
                continue
            end_path = site_path / "end.txt"
            if not end_path.exists():
                continue

            save_path = save_root / label_folder / f"{site}.pt"
            all_items.append((int(label_idx), end_path, save_path))

    # Load encoder
    if backend == "hf_text":
        tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=False)
        model = AutoModel.from_pretrained(hf_id).to(DEVICE)
        model.eval()
    else:  # KoCLIP text branch
        processor = AutoProcessor.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id).to(DEVICE)  # returns a CLIPModel
        model.eval()

    # Process files
    for label_idx, end_path, save_path in tqdm(all_items, desc=f"📄 [{model_key}] Embedding end.txt"):
        # Read file
        try:
            html = end_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"❌ read error: {end_path} → {e}")
            with open(PARSER_ERR_LOG, "a", encoding="utf-8") as logf:
                logf.write(f"[READ]\t{end_path}\n")
            continue

        # HTML → text
        try:
            text = extract_text_from_html(html)
        except Exception as e:
            print(f"❌ parse error: {end_path} → {e}")
            with open(PARSER_ERR_LOG, "a", encoding="utf-8") as logf:
                logf.write(f"[PARSE]\t{end_path}\n")
            continue

        # Encode & save
        try:
            if backend == "hf_text":
                enc = tokenizer(
                    text,
                    max_length=max_len,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                with torch.no_grad():
                    outputs = model(**enc)
                    last_hidden = outputs.last_hidden_state  # (1, L, H)
                    sent = cls_pool(last_hidden) if pooling == "cls" else mean_pool(last_hidden, enc["attention_mask"])
                    sent_emb = sent.squeeze(0).cpu()  # (H,)
            else:
                if len(text) > 512:
                    text = text[:512]  # ⚠️ Cut off too long text

                inputs = processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}  # ✅ send to GPU
                
                with torch.no_grad():
                    outputs = model.get_text_features(**inputs)
                    sent_emb = outputs.squeeze(0).cpu() # save in CPU
            # Save tensor
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"text_emb": sent_emb, "label": int(label_idx)}, save_path)

        except Exception as e:
            print(f"❌ embed error: {end_path} → {e}")
            with open(PARSER_ERR_LOG, "a", encoding="utf-8") as logf:
                logf.write(f"[EMBED]\t{end_path}\n")
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, choices=list(MODEL_PRESETS.keys()),
                    help="Encoder key: koelectra | kcbert | koclip")
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"],
                    help="Sentence pooling (mean or cls)")
    ap.add_argument("--max_len", type=int, default=512,
                    help="Max token length (only used for hf_text backends)")
    args = ap.parse_args()

    print(f"[INFO] model={args.model}, pooling={args.pooling}, max_len={args.max_len}, device={DEVICE}")
    preprocess_all(model_key=args.model, pooling=args.pooling, max_len=args.max_len)
    print("[DONE] saved under:", PT_DIR / f"{args.model}_pt_end")


if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    print(f"[TIME] Elapsed: {t1 - t0:.2f} seconds")

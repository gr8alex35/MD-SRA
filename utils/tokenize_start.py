# utils/tokenize_end.py
# -*- coding: utf-8 -*-
"""
Build text embeddings for start.txt using selectable encoders:
- HF text encoders (AutoTokenizer + AutoModel)
- KoCLIP (AutoProcessor + CLIPModel.get_text_features)

Saves tensors under: pt_data/<model_key>_pt_start/<label>/<site>.pt

Saved file content:
{
  "text_emb": torch.Tensor(H),   # sentence embedding
  "feature":  torch.Tensor(1, 9),# side features (redirect/script signals)
  "label":    int                # class index from folder name prefix
}
"""
import os
import re
import time
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel, AutoProcessor

IMPORTANT_TAGS = ['title', 'h1', 'h2', 'h3', 'h4', 'td', 'p', 'a', 'b', 'span', 'button', 'li']

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
#   backend: "hf_text" | "koclip"
# -----------------------------
MODEL_PRESETS = {
    "koelectra": ("hf_text", "monologg/koelectra-base-v3-discriminator", "KoELECTRA base"),
    "kcbert":    ("hf_text", "beomi/KcBERT-base",                        "KcBERT base"),
    "koclip":    ("koclip",  "koclip/koclip-base-pt",                    "KoCLIP base (text)"),
}

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================
# HTML → clean text
# =============================
def clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_html(html: str, max_char_len: int = 4096) -> str:
    # prefer lxml, fallback to std parser
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
# 9-dim redirect/script feature
# =============================
def extract_feature(html: str) -> torch.Tensor:
    soup = BeautifulSoup(html, 'html.parser')
    script_tags = soup.find_all('script')
    meta_tags = soup.find_all('meta')
    script_texts = [tag.get_text() for tag in script_tags if tag.get_text()]

    full_script = "\n".join(script_texts)
    has_meta_refresh = int(any(
        ("http-equiv" in tag.attrs) and ("refresh" in str(tag.attrs.get("http-equiv", "")).lower())
        for tag in meta_tags
    ))

    feat = torch.tensor([
        int("window.location" in full_script),
        int("document.location" in full_script),
        int("top.location" in full_script),
        int("window.open" in full_script),
        int("eval(" in full_script),
        has_meta_refresh,
        len(script_tags),
        max([len(s) for s in script_texts], default=0) / 1000.0,
        (
            full_script.count(')') + full_script.count('}') + full_script.count('=') +
            len(re.findall(r'atob\s*\(', full_script)) +
            len(re.findall(r'btoa\s*\(', full_script)) +
            len(re.findall(r'unescape\s*\(', full_script)) +
            len(re.findall(r'eval\(function\(p,a,c,k,e,r\)', full_script)) +
            len([w for w in full_script.split() if len(w) >= 512])
        ) / 100.0
    ], dtype=torch.float)
    return feat  # shape: [9]


# =============================
# Pooling (HF text encoders)
# =============================
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom  # (B, H)

def cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, 0, :]  # (B, H)


# =============================
# Main
# =============================
def preprocess_all(model_key: str, pooling: str = "mean", max_len: int = 512):
    assert model_key in MODEL_PRESETS, f"Unknown model_key: {model_key}"
    backend, hf_id, _ = MODEL_PRESETS[model_key]

    # Load encoder(s)
    if backend == "hf_text":
        tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=False)
        model = AutoModel.from_pretrained(hf_id).to(DEVICE)
        model.eval()
    else:
        processor = AutoProcessor.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id).to(DEVICE)  # CLIPModel
        model.eval()
        CLIP_MAX_TOK = getattr(getattr(model.config, "text_config", None), "max_position_embeddings", 77)

    # Save directory
    save_root = PT_DIR / f"{model_key}_pt_start"
    os.makedirs(save_root, exist_ok=True)

    # Reset parser error log
    open(PARSER_ERR_LOG, "w").close()

    # Collect file list
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
            start_path = site_path / "start.txt"
            if not start_path.exists():
                continue

            save_path = save_root / label_folder / f"{site}.pt"
            all_items.append((int(label_idx), start_path, save_path))

    # Process
    for label_idx, start_path, save_path in tqdm(all_items, desc=f"📄 [{model_key}] Embedding start.txt"):
        # Read HTML
        try:
            html = start_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"❌ read error: {start_path} → {e}")
            with open(PARSER_ERR_LOG, "a", encoding="utf-8") as logf:
                logf.write(f"[READ]\t{start_path}\n")
            continue

        # Extract visible text
        try:
            text = extract_text_from_html(html)
        except Exception as e:
            print(f"❌ parse error: {start_path} → {e}")
            with open(PARSER_ERR_LOG, "a", encoding="utf-8") as logf:
                logf.write(f"[PARSE]\t{start_path}\n")
            continue

        try:
            # --- Encode ---
            if backend == "hf_text":
                enc = {
                    k: v.to(DEVICE) for k, v in AutoTokenizer.from_pretrained(hf_id, use_fast=False)(
                        text,
                        max_length=max_len,
                        truncation=True,
                        padding=True,
                        return_tensors="pt"
                    ).items()
                }
                with torch.no_grad():
                    outputs = model(**enc)
                    last_hidden = outputs.last_hidden_state  # (1, L, H)
                    sent_emb = cls_pool(last_hidden) if pooling == "cls" else mean_pool(last_hidden, enc["attention_mask"])
                    sent_emb = sent_emb.squeeze(0).cpu()  # (H,)
            else:
                # KoCLIP in the same style you used for end:
                if len(text) > 512:
                    text = text[:512]  # simple character cut to avoid extreme cases

                inputs = processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=CLIP_MAX_TOK,  # e.g., 77
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

                with torch.no_grad():
                    outputs = model.get_text_features(**inputs)  # (1, H_t)
                    sent_emb = outputs.squeeze(0).cpu()

            # Side feature (9-dim)
            feature = extract_feature(html).unsqueeze(0)  # (1, 9)

            # Save
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"text_emb": sent_emb, "feature": feature, "label": int(label_idx)}, save_path)

        except Exception as e:
            print(f"❌ embed error: {start_path} → {e}")
            with open(PARSER_ERR_LOG, "a", encoding="utf-8") as logf:
                logf.write(f"[EMBED]\t{start_path}\n")
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, choices=list(MODEL_PRESETS.keys()),
                    help="Encoder key: koelectra | kcbert | koclip")
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"],
                    help="Pooling for hf_text backends (ignored for koclip)")
    ap.add_argument("--max_len", type=int, default=512, help="Max token length for hf_text backends")
    args = ap.parse_args()

    print(f"[INFO] model={args.model}, pooling={args.pooling}, max_len={args.max_len}, device={DEVICE}")
    preprocess_all(model_key=args.model, pooling=args.pooling, max_len=args.max_len)
    print("[DONE] all saved under:", PT_DIR / f"{args.model}_pt_start")


if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    print(f"[TIME] Elapsed time: {t1 - t0:.2f} seconds")

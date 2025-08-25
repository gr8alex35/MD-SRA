# utils/tokenize_img_generic.py
# -*- coding: utf-8 -*-
import os
import argparse
from pathlib import Path
import time
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from transformers import AutoProcessor, AutoModel
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# =========================
# Device
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Encoders
# =========================
class KoCLIPImageEncoder:
    def __init__(self, model_name="koclip/koclip-base-pt"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.out_dim = 512

    @torch.no_grad()
    def encode(self, img: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        feat = self.model.get_image_features(**inputs)  # (1, 512)
        return feat.squeeze(0).cpu()


class TimmEncoder:
    """Generic image encoder using timm (ViT / ResNet). Global-average-pools to a vector."""
    def __init__(self, model_name: str):
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 -> feature vector
        self.model = self.model.to(DEVICE).eval()
        self.cfg = resolve_data_config({}, model=self.model)
        self.tf = create_transform(**self.cfg)
        # 추론하여 out_dim 추정
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.cfg.get("input_size", (3, 224, 224))[1], self.cfg.get("input_size", (3, 224, 224))[2]).to(DEVICE)
            out = self.model(dummy)
            self.out_dim = out.shape[-1]

    @torch.no_grad()
    def encode(self, img: Image.Image) -> torch.Tensor:
        x = self.tf(img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
        feat = self.model(x).squeeze(0).detach().cpu()  # (D,)
        return feat


# =========================
# I/O helpers
# =========================
def load_image(path: Path) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except (FileNotFoundError, UnidentifiedImageError) as e:
        raise RuntimeError(f"cannot open image: {path} -> {e}")

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=str, default="koclip",
                    choices=["koclip", "vit", "resnet"],
                    help="Select image encoder (koclip/koclip-base-pt, timm vit_base, timm resnet50)")
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent.parent / "data"),
                    help="Data path (label/site 폴더 구조)")
    ap.add_argument("--image_name", type=str, default="end_screenshot.png",
                    help="Name of the image in each folder")
    ap.add_argument("--out_root", type=str, default=str(Path(__file__).resolve().parent.parent / "pt_data"),
                    help="Saving Path ({encoder}_pt_image")
    ap.add_argument("--vit_name", type=str, default="vit_base_patch16_224",
                    help="timm ViT Model Name")
    ap.add_argument("--resnet_name", type=str, default="resnet50",
                    help="timm ResNet Model Name")
    args = ap.parse_args()

    data_root = Path(args.root)
    out_root = Path(args.out_root) / f"{args.encoder}_pt_image"
    out_root.mkdir(parents=True, exist_ok=True)

    # encoder 선택
    if args.encoder == "koclip":
        encoder = KoCLIPImageEncoder()
    elif args.encoder == "vit":
        encoder = TimmEncoder(args.vit_name)
    else:  # resnet
        encoder = TimmEncoder(args.resnet_name)

    print(f"[INFO] encoder={args.encoder} (out_dim≈{encoder.out_dim}), device={DEVICE}")
    print(f"[INFO] input_root={data_root}, out_root={out_root}")

    folders = [f for f in (data_root).glob("*_*/*") if f.is_dir()]
    print(f"📁 Total site folders: {len(folders)}")

    for folder in tqdm(folders, desc=f"🖼️ Embedding images with {args.encoder}"):
        name = folder.name
        label_folder = folder.parent.name
        image_path = folder / args.image_name

        if not image_path.exists():
            # print(f"⚠️ {name}: {args.image_name} doesn't exist → Skip")
            continue

        try:
            img = load_image(image_path)
            emb = encoder.encode(img)  # (D,)
        except Exception as e:
            print(f"❌ {name}: {e}")
            continue

        save_dir = out_root / label_folder
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(emb, save_dir / f"{name}.pt")

    print("✅ Done.")

if __name__ == "__main__":
    start_time = time.time()   # Starting Time
    main()
    end_time = time.time()     # Finishing Time
    elapsed = end_time - start_time
    print(f"[TIME] Elapsed time: {elapsed:.2f} seconds")


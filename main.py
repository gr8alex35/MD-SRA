import os
import time
import torch
import argparse
from torch.utils.data import DataLoader, Subset
from transformers import logging as transformers_logging
from utils.dataset import MultimodalDataset
from utils.models import MDSRA
from utils.metrics import evaluate

transformers_logging.set_verbosity_error()

def _infer_encoder_names(start_dir: str | None, end_dir: str | None, image_dir: str | None,
                         use_start: bool, use_end: bool, use_image: bool) -> dict:
    """
    Infer encoder names from pt_data directory names.
    Returns e.g., {"text": "kcbert", "image": "resnet"}.
    """
    def parse_from_path(path: str, suffix: str) -> str | None:
        if not path:
            return None
        base = os.path.basename(path.rstrip("/"))
        if base.endswith(suffix):
            return base[:-len(suffix)]
        if "_pt_" in base:
            return base.split("_pt_")[0]
        return base

    encoders = {}

    # Text encoder for S/E/SE
    if use_start or use_end:
        enc = parse_from_path(start_dir, "_pt_start") or parse_from_path(end_dir, "_pt_end")
        if enc:
            encoders["text"] = enc

    # Image encoder
    if use_image:
        enc = parse_from_path(image_dir, "_pt_image")
        if enc:
            encoders["image"] = enc

    return encoders if encoders else {"text": "model"}

def build_model_key(start_dir, end_dir, image_dir, use_start, use_end, use_image) -> str:
    """
    Build a human-readable model key based on enabled modalities and inferred encoder names.
    """
    encoders = _infer_encoder_names(start_dir, end_dir, image_dir, use_start, use_end, use_image)
    mods = []
    if use_start:
        mods.append("S")
    if use_end:
        mods.append("E")
    if use_image:
        mods.append("I")

    # SE + I -> {text_enc}_SE_{image_enc}_I
    if use_start and use_end and use_image:
        return f"{encoders.get('text', 'model')}_SE_{encoders.get('image', 'model')}_I"

    # S + I or E + I -> {text_enc}_{S|E}_{image_enc}_I
    if use_image and (use_start ^ use_end):  # exactly one of S/E
        se_tag = "S" if use_start else "E"
        return f"{encoders.get('text', 'model')}_{se_tag}_{encoders.get('image', 'model')}_I"

    # Text-only -> {text_enc}_SE / {text_enc}_S / {text_enc}_E
    if (use_start or use_end) and not use_image:
        return f"{encoders.get('text', 'model')}_{'_'.join(mods)}"

    # Image-only -> {image_enc}_I
    if use_image and not (use_start or use_end):
        return f"{encoders.get('image', 'model')}_I"

    return "model_none"


# -----------------------------
# Stratified split utilities
# -----------------------------
def _collect_labels(dataset):
    """
    Extract labels from the dataset once to avoid repeated indexing overhead.
    Assumes each dataset item is a dict with key 'label'.
    """
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(int(item["label"]))
    return labels

def stratified_split(dataset, splits=(0.6, 0.2, 0.2), seed: int = 42):
    """
    Perform a stratified split (train/val/test) without external dependencies.
    - Keeps class distribution approximately the same across splits.
    - Returns three Subset objects.
    """
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    gen = torch.Generator()
    gen.manual_seed(seed)

    labels = _collect_labels(dataset)
    num_classes = max(labels) + 1  # assumes labels are 0..C-1
    per_class_indices = {c: [] for c in range(num_classes)}
    for idx, y in enumerate(labels):
        per_class_indices[y].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for c, idxs in per_class_indices.items():
        # Shuffle per-class indices deterministically
        idxs_tensor = torch.tensor(idxs)
        perm = torch.randperm(len(idxs_tensor), generator=gen)
        idxs_shuffled = idxs_tensor[perm].tolist()

        n = len(idxs_shuffled)
        n_train = int(round(n * splits[0]))
        n_val = int(round(n * splits[1]))
        # Ensure all samples are assigned
        n_test = n - n_train - n_val
        if n_test < 0:
            # Adjust (rare rounding case)
            n_test = max(0, n - n_train)
            n_val = n - n_train - n_test

        train_idx.extend(idxs_shuffled[:n_train])
        val_idx.extend(idxs_shuffled[n_train:n_train + n_val])
        test_idx.extend(idxs_shuffled[n_train + n_val:])

    # Final shuffle of each split for randomness
    def _shuffle(lst):
        t = torch.tensor(lst)
        perm = torch.randperm(len(t), generator=gen)
        return t[perm].tolist()

    train_subset = Subset(dataset, _shuffle(train_idx))
    val_subset = Subset(dataset, _shuffle(val_idx))
    test_subset = Subset(dataset, _shuffle(test_idx))
    return train_subset, val_subset, test_subset


# -----------------------------
# Training configurations
# -----------------------------
best_val_acc = 0
patience, patience_limit = 0, 50
n_class = 5


def train_model(start_dir, end_dir, image_dir, use_start, use_end, use_image, device, log_path=None):
    """
    Train MDRIOG with stratified train/val/test splits and cosine LR schedule.
    Saves the best checkpoint by validation accuracy and performs a final test.
    """
    global best_val_acc, patience, patience_limit, n_class

    # Build a unique model key for logs/checkpoints
    model_key = build_model_key(
        start_dir=start_dir,
        end_dir=end_dir,
        image_dir=image_dir,
        use_start=use_start,
        use_end=use_end,
        use_image=use_image,
    )

    # Prepare log file
    os.makedirs("logs", exist_ok=True)
    if log_path is None:
        log_path = f"logs/ablation_{model_key}.log"
    logf = open(log_path, 'w')

    def logprint(msg):
        print(msg)
        logf.write(msg + '\n')

    # Timestamp: training start
    start_time = time.time()
    logprint(f"[INFO] Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    # Load dataset and make stratified splits (60/20/20)
    dataset = MultimodalDataset(start_dir, end_dir, image_dir, n_class=n_class)
    train_set, val_set, test_set = stratified_split(dataset, splits=(0.6, 0.2, 0.2), seed=42)

    loader_train = DataLoader(train_set, batch_size=64, shuffle=True)
    loader_val = DataLoader(val_set, batch_size=64)
    loader_test = DataLoader(test_set, batch_size=64)

    # Model / optimizer / loss / scheduler
    model = MDSRA(use_start, use_end, use_image, n_class=n_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=300)

    # Checkpoint path
    os.makedirs("checkpoints", exist_ok=True)
    best_path = f"checkpoints/{model_key}.pt"

    for epoch in range(1, 301):
        model.train()
        total_loss = 0.0

        for batch in loader_train:
            inputs = {}
            if use_start:
                inputs["start_text"] = batch["start"]["text_emb"].to(device)
                inputs["start_feat"] = batch["start"]["feature"].to(device)
            if use_end:
                inputs["end_text"] = batch["end"].to(device)
            if use_image:
                inputs["image_emb"] = batch["image"].to(device)

            labels = batch["label"].to(device)
            outputs = model(**inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Step the cosine scheduler after each epoch
        scheduler.step()

        val_acc = evaluate(model, loader_val, device, n_class)
        logprint(f"[Epoch {epoch}] Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}")

        global best_val_acc, patience
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), best_path)
            logprint("✅ Best model updated")
        else:
            patience += 1
            if patience >= patience_limit:
                logprint("⏹ Early stopping triggered")
                break

    # Timestamp: training end
    end_time = time.time()
    elapsed = end_time - start_time
    logprint(f"[INFO] Training finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    logprint(f"[INFO] Total training time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    # Final evaluation on the stratified test set using the best checkpoint
    model.load_state_dict(torch.load(best_path))
    logprint("\n📊 Final Evaluation")
    eval_start = time.time()
    _ = evaluate(model, loader_test, device, n_class, detailed=True, logf=logf)
    eval_end = time.time()
    eval_elapsed = eval_end - eval_start

    logprint(f"[INFO] Evaluation finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eval_end))}")
    logprint(f"[INFO] Total evaluation time: {eval_elapsed:.2f} seconds ({eval_elapsed/60:.2f} minutes)")
    logf.close()

def eval_model(start_dir, end_dir, image_dir, use_start, use_end, use_image, device):
    """
    Load the best checkpoint and evaluate on a stratified test split (60/20/20 from the same seed).
    """
    model_key = build_model_key(
        start_dir=start_dir,
        end_dir=end_dir,
        image_dir=image_dir,
        use_start=use_start,
        use_end=use_end,
        use_image=use_image,
    )

    dataset = MultimodalDataset(start_dir, end_dir, image_dir, n_class=n_class)
    _, _, test_set = stratified_split(dataset, splits=(0.6, 0.2, 0.2), seed=42)
    loader_test = DataLoader(test_set, batch_size=64)

    model = MDSRA(use_start, use_end, use_image, n_class=n_class).to(device)
    best_path = f"checkpoints/{model_key}.pt"
    assert os.path.exists(best_path), f"Model checkpoint not found at {best_path}"
    model.load_state_dict(torch.load(best_path))

    print(f"\n📊 Evaluation for {model_key}")
    _ = evaluate(model, loader_test, device, n_class, detailed=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_dir', type=str, default="./pt_data/koclip_pt_start")
    parser.add_argument('--end_dir', type=str, default="./pt_data/koclip_pt_end")
    parser.add_argument('--image_dir', type=str, default="./pt_data/koclip_pt_image")
    parser.add_argument('--use_start', action='store_true')
    parser.add_argument('--use_end', action='store_true')
    parser.add_argument('--use_image', action='store_true')
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true', help="Only evaluate the saved best model")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        eval_model(
            start_dir=args.start_dir,
            end_dir=args.end_dir,
            image_dir=args.image_dir,
            use_start=args.use_start,
            use_end=args.use_end,
            use_image=args.use_image,
            device=device
        )
    else:
        train_model(
            start_dir=args.start_dir,
            end_dir=args.end_dir,
            image_dir=args.image_dir,
            use_start=args.use_start,
            use_end=args.use_end,
            use_image=args.use_image,
            device=device,
            log_path=args.log_path
        )

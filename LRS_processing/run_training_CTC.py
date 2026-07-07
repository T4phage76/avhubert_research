import torch
import random
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from torch.nn import CTCLoss
from torch import optim
from LRS_processing.build_dataloader import AVSRDataset,avsr_collate_fn
from LRS_processing.build_model import CrossAV
from LRS_processing.training_frameCE import train
import json
import os
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Subset
import shutil, time



def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple chips
    elif torch.cuda.is_available():
        return torch.device("cuda")  # Nvidia CUDA GPU
    else:
        return torch.device("cpu")  # Fallback
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def my_collate(batch):
    return avsr_collate_fn(batch, ctc_filter=False)



# ---- (1) Repro & device ----
# set seed for everything 
set_seed(seed = 20250817)
torch.use_deterministic_algorithms(True)

# device = get_device()
device = torch.device("cpu")


# ---- (2) Datasets & DataLoaders ----
# get phone to index mapping 
phone_to_idx_path="LRS_processing/LRS2/LRS_main_labelled_data/cmu_phoneme_to_idx.json"
with open(phone_to_idx_path, 'r') as f:
    phoneme_to_idx = json.load(f)

# read
train_dataset = torch.load("LRS_processing/LRS2/LRS2_preprocessed_datasets/Aug16_train_dataset(main).pt")
val_dataset = torch.load("LRS_processing/LRS2/LRS2_preprocessed_datasets/Aug16_validation_dataset(main).pt")
test_dataset = torch.load("LRS_processing/LRS2/LRS2_preprocessed_datasets/Aug16_test_dataset(main).pt")


train_loader = DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=False,
    num_workers=0,  # or 0 ?
    collate_fn=my_collate,
    worker_init_fn=seed_worker,
    drop_last=False
    )
val_loader = DataLoader(
    val_dataset, 
    batch_size=4, 
    shuffle=False,
    num_workers=0,  # or 0 ?
    collate_fn=my_collate,
    worker_init_fn=seed_worker,
    drop_last=False
    )
test_loader = DataLoader(
    test_dataset, 
    batch_size=4, 
    shuffle=False,
    num_workers=0,  # or 0 ?
    collate_fn=my_collate,
    worker_init_fn=seed_worker,
    drop_last=False
    )


import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import logging


# --------- Setup logging ---------

# --- Training logger ---
logging.captureWarnings(True)

train_handler = logging.FileHandler("training_warnings.log", mode="w")
train_handler.setLevel(logging.INFO)  # or INFO if you want everything
train_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

train_logger = logging.getLogger("train_logger")
train_logger.setLevel(logging.INFO)
train_logger.propagate = False  
train_logger.addHandler(train_handler)

pywarn_logger_train = logging.getLogger("py.warnings")
pywarn_logger_train.setLevel(logging.WARNING)     # warnings module sends WARNING records
pywarn_logger_train.propagate = False
pywarn_logger_train.addHandler(train_handler)

# also echo to console
# train_stream = logging.StreamHandler()
# train_stream.setFormatter(train_formatter)
# train_logger.addHandler(train_stream)


# --- Validation logger ---
val_handler = logging.FileHandler("validation_warnings.log", mode="w")
val_handler.setLevel(logging.WARNING)
val_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

val_logger = logging.getLogger("val_logger")
val_logger.setLevel(logging.INFO)
val_logger.propagate = False  
val_logger.addHandler(val_handler)

# val_stream = logging.StreamHandler()
# val_stream.setFormatter(val_formatter)
# val_logger.addHandler(val_stream)


# ---- (3) Model ----
def _safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def _ckpt_name(ckpt_dir, epoch):
    return os.path.join(ckpt_dir, f"epoch{epoch:03d}.pt")

def _save_checkpoint(path, model, optimizer, scaler, epoch, val_loss, extra=None):
    payload = {
        "epoch": epoch,
        "val_loss": float(val_loss),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": (scaler.state_dict() if scaler is not None and scaler.is_enabled() else None),
        "extra": extra or {},
        "timestamp": time.time(),
        "pytorch_version": torch.__version__,
    }
    torch.save(payload, path)

def _copy_file(src, dst):
    shutil.copy2(src, dst)

def train_ctc(
    # training data
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion, 
    device,
    #training parameters
    epochs=20, 
    patience=5, 
    grad_clip=1.0, 
    log_interval=40, 
    use_amp=True,
    # checkpointing:
    ckpt_dir="LRS_processing/LRS2/checkpoints/Cross_attn_test", # dir t osave the checkpoints
    log_json="train_log.json" # json to savethe training outputs (loss, epoch etc)
):
    _safe_mkdir(ckpt_dir)

    best_val = float('inf')
    best_epoch = -1
    no_improve = 0

    # list to store training process
    log_history = []

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and torch.cuda.is_available()))
    

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        steps = 0

        for step, batch in enumerate(train_loader, 1):
            # collate may return None if all items were filtered
            if batch is None:
                train_logger.warning(f"Empty batch at step {step},epoch {epoch}, skipping...")
                continue

            audio = batch['audio'].to(device)              # (B, T_a_max, F)
            video = batch['video'].to(device)              # (B, T_v_max, H, W)
            targets = batch['labels'].to(device)           # (sum_target,)
            in_lens = batch.get('input lengths').to(device).long()     # (B,)
            tgt_lens = batch.get('target lengths').to(device).long() # (B,)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=(use_amp and torch.cuda.is_available())):
                if model.return_attn:
                    logits, attn = model(
                        audio_tensor=audio, visual_tensor=video, mode="av",
                        lengths_audio=in_lens, lengths_video=in_lens, attn_mask=None
                    )   # (B, T, C)
                else:
                    logits = model(
                        audio_tensor=audio, visual_tensor=video, mode="av",
                        lengths_audio=in_lens, lengths_video=in_lens, attn_mask=None
                    )

                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)   # (T, B, C)
                train_batch_loss = criterion(log_probs, targets, in_lens, tgt_lens)

                # non-finite guard (train)
                if not torch.isfinite(train_batch_loss):
                    # save a crash checkpoint before exiting
                    crash_path = os.path.join(ckpt_dir, "crash.pt")
                    torch.save({
                        "epoch": epoch,
                        "step": step,
                        "reason": "non-finite loss",
                        "loss": float(train_batch_loss.detach().cpu().item() if train_batch_loss.numel() == 1 else float('nan')),
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": (scaler.state_dict() if scaler is not None and scaler.is_enabled() else None),
                    }, crash_path)
                    train_logger.error(f"Non-finite loss at epoch {epoch}, step {step}. Saved {crash_path}.")
                    return {"best_val": float(best_val), "best_epoch": int(best_epoch)}

            # backward + step
            if scaler.is_enabled():
                scaler.scale(train_batch_loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                train_batch_loss.backward()
                if grad_clip is not None:
                    clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            train_loss_accum += float(train_batch_loss.item())
            steps += 1
            train_running_loss = train_loss_accum / max(1, steps)

            if step % log_interval == 0:
                print(f"[Epoch {epoch} | Step {step}/{len(train_loader)}] "
                      f"Train CTC loss: {train_running_loss:.4f}")
                
            log_history.append({"phase": "train",
                                "epoch": epoch, 
                                "step": step, 
                                "running_loss": train_running_loss, 
                                "batch_loss": float(train_batch_loss.item()),})
            


        # --------- Validation ---------
        model.eval()
        val_loss_accum = 0.0
        vsteps = 0

        with torch.no_grad():
            for vstep, batch in enumerate(val_loader, 1):
                if batch is None:
                    val_logger.warning(f"Empty batch at val step {vstep}, epoch {epoch}, skipping...")
                    continue

                audio = batch['audio'].to(device)
                video = batch['video'].to(device)
                targets = batch['labels'].to(device)
                in_lens = batch.get('input lengths').to(device).long()
                tgt_lens = batch.get('target lengths').to(device).long()

                if model.return_attn:
                    logits, _ = model(
                        audio_tensor=audio, visual_tensor=video, mode="av",
                        lengths_audio=in_lens, lengths_video=in_lens, attn_mask=None
                    )
                else:
                    logits = model(
                        audio_tensor=audio, visual_tensor=video, mode="av",
                        lengths_audio=in_lens, lengths_video=in_lens, attn_mask=None
                    )

                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)   # (T,B,C)
                val_batch_loss = criterion(log_probs, targets, in_lens, tgt_lens)

                if not torch.isfinite(val_batch_loss):
                    crash_path = os.path.join(ckpt_dir, "crash_val.pt")
                    torch.save({
                        "epoch": epoch, "step": vstep, "reason": "non-finite val loss",
                        "loss": float(val_batch_loss.detach().cpu().item() if val_batch_loss.numel() == 1 else float('nan')),
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": (scaler.state_dict() if scaler is not None and scaler.is_enabled() else None),
                        "timestamp": time.time(),
                    }, crash_path)
                    val_logger.error(f"Non-finite VAL loss at epoch {epoch}, step {vstep}. Saved {crash_path}.")
                    return {"best_val": float(best_val), "best_epoch": int(best_epoch)}

                val_loss_accum += float(val_batch_loss.item())
                vsteps += 1

        val_running_loss = val_loss_accum / max(1, vsteps)
        print(f"==> Epoch {epoch}: Train {train_running_loss:.4f} | Val {val_running_loss:.4f}")

        log_history.append({
            "phase": "validation",
            "epoch": epoch,
            "train_loss": float(train_running_loss),
            "val_loss": float(val_running_loss),
        })


        # --------- Save checkpoints ---------
        last_path = _ckpt_name(ckpt_dir, epoch)
        _save_checkpoint(
            last_path, model, optimizer, scaler, epoch, val_running_loss,
            extra={"train_loss": float(train_running_loss)}
        )

        # update best if improved
        if val_running_loss < best_val:
            best_val = val_running_loss
            best_epoch = epoch
            _copy_file(last_path, os.path.join(ckpt_dir, "best.pt"))
            no_improve = 0
        else:
            no_improve += 1
            print(f"   no improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print("Early stopping.")
                break

        # persist logs each epoch
        with open(os.path.join(ckpt_dir, log_json), "w") as f:
            json.dump(log_history, f, indent=2)


    # --------- Restore best weights ---------
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        best_payload = torch.load(best_path, map_location="cpu")
        model.load_state_dict(best_payload["model_state"])
    else:
        print("Warning: best.pt not found; leaving model at last epoch weights.")


    # final log save
    with open(os.path.join(ckpt_dir, log_json), "w") as f:
        json.dump(log_history, f, indent=2)

    return {"best_val": float(best_val), "best_epoch": int(best_epoch)}

# Make sure CrossAV, AuditoryEncoder, VisualEncoder, CrossAttentionFusion are imported/defined.
model = CrossAV(
    phoneme_vocab_size=42,
    a_drop_modality_prob=0.3,
    v_drop_modality_prob=0.1,
    return_attn=True,                # keep attention for optional analysis
    fusion_mask_type="causal_band",  # your choice
    fusion_band=5,
    use_q_proj=True,
    use_k_proj=True,
    use_v_proj=True,
).to(device)

# ---- (4) Criterion & Optimizer ----
# NOTE: blank index must match your label space (0 is common)
criterion = CTCLoss(blank=0, reduction="mean", zero_infinity=True)

# A good starting optimizer for this kind of model
optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
)

# ---- (5) Checkpoint dir ----
ckpt_dir = "LRS_processing/LRS2/checkpoints/Cross_attn_test"
os.makedirs(ckpt_dir, exist_ok=True)

# ---- (6) Train! ----

summary = train_ctc(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    epochs=20,
    patience=5,
    grad_clip=1.0,
    log_interval=40,
    use_amp=False,         # automatically diable on CPU               
    ckpt_dir=ckpt_dir,
    log_json="train_log.json",
)

print("Training summary:", summary)


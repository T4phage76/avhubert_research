import torch
import random
import numpy as np
from torch import optim
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from LRS_processing.build_dataloader import AVSRDataset,avsr_collate_fn
from LRS_processing.build_model import CrossAV
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Subset
import os, time, shutil, json
from torch.nn import CTCLoss
from tqdm import tqdm

from torch.nn.utils import clip_grad_norm_
import logging



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


def build_valid_index_list(dataset):
    valid_idxs = []
    invalid = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        in_len  = sample["video"].shape[0]
        tgt_len = len(sample["labels"])
        if in_len >= 2 * tgt_len - 1:
            valid_idxs.append(i)
        else:
            invalid += 1
    print(f"[prefilter] kept {len(valid_idxs)} / {len(dataset)} (dropped {invalid})")
    return valid_idxs

# set seed for everything 
set_seed(seed = 20250826)
# torch.use_deterministic_algorithms(True)
# device = get_device()
device = torch.device("cpu")


# get phone to index mapping 
phone_to_idx_path="LRS_processing/LRS2/LRS_main_labelled_data/cmu_phoneme_to_idx.json"
with open(phone_to_idx_path, 'r') as f:
    phoneme_to_idx = json.load(f)

# run data set and split to train and validation sets 
# dataset = AVSRDataset(
#     audio_dir="LRS_processing/LRS2/LRS_main_audio",
#     video_dir="LRS_processing/LRS2/LRS_main_mouth_roi",
#     label_json_path="LRS_processing/LRS2/LRS_main_labelled_data/main_labels_whole_clip_cmu.json",
#     phoneme_to_idx = phoneme_to_idx,
#     sample_rate=16000
# )
# valid_dataset = Subset(dataset, build_valid_index_list(dataset))

# total_len = len(valid_dataset)
# train_len = int(0.6 * total_len)
# val_len = (total_len - train_len)//2

# train_dataset, val_dataset, test_dataset = random_split(valid_dataset, [train_len, val_len, total_len-train_len-val_len])

# torch.save(valid_dataset, "LRS_processing/LRS2/LRS2_preprocessed_datasets/Aug16_preprocessed_dataset(all main).pt")
# torch.save(train_dataset, "LRS_processing/LRS2/LRS2_preprocessed_datasets/Aug16_train_dataset(main).pt")
# torch.save(val_dataset, "LRS_processing/LRS2/LRS2_preprocessed_datasets/Aug16_validation_dataset(main).pt")
# torch.save(test_dataset, "LRS_processing/LRS2/LRS2_preprocessed_datasets/Aug16_test_dataset(main).pt")
# read
train_dataset = torch.load("LRS_processing/LRS2/LRS2_preprocessed_datasets/Aug16_train_dataset(main).pt", weights_only=False)
val_dataset = torch.load("LRS_processing/LRS2/LRS2_preprocessed_datasets/Aug16_validation_dataset(main).pt",weights_only=False)
test_dataset = torch.load("LRS_processing/LRS2/LRS2_preprocessed_datasets/Aug16_test_dataset(main).pt",weights_only=False)

n_train_subset = 3000
n_val_subset   = 600   

rng = np.random.default_rng(20250823)  # fixed seed for reproducibility
train_indices = rng.choice(len(train_dataset), size=n_train_subset, replace=False)
val_indices   = rng.choice(len(val_dataset), size=n_val_subset, replace=False)

train_subset = Subset(train_dataset, train_indices)
val_subset   = Subset(val_dataset, val_indices)

def my_collate(batch):
    return avsr_collate_fn(batch, ctc_filter=False)

train_loader = DataLoader(
    train_dataset, 
    batch_size=20, 
    shuffle=False,
    num_workers=0,  # or 0 ?
    collate_fn=my_collate,
    worker_init_fn=seed_worker,
    drop_last=False
    )
val_loader = DataLoader(
    val_dataset, 
    batch_size=20, 
    shuffle=False,
    num_workers=0,  # or 0 ?
    collate_fn=my_collate,
    worker_init_fn=seed_worker,
    drop_last=False
    )
test_loader = DataLoader(
    test_dataset, 
    batch_size=20, 
    shuffle=False,
    num_workers=0,  # or 0 ?
    collate_fn=my_collate,
    worker_init_fn=seed_worker,
    drop_last=False
    )

# test dataset
first_batch = next(iter(train_loader))
first_batch['input lengths'] # real input length (without padding)
first_batch['target lengths']
first_batch['labels']
first_batch['input length'][2]
first_batch['video'][2][69,:,:]
first_batch.get('input length',first_batch.get('input length' ) )


# --------- tiny helper for greedy CTC decode ----------
def _ctc_greedy_decode(logits, blank_idx=0):
    """
    logits: (B, T, C) BEFORE softmax.
    Returns: list of 1D LongTensor with collapsed predictions per batch item.
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)  # (B, T)
        decoded = []
        for seq in preds:  # (T,)
            out = []
            prev = None
            for p in seq.tolist():
                if p != blank_idx and p != prev:
                    out.append(p)
                prev = p
            decoded.append(torch.tensor(out, dtype=torch.long))
        return decoded

# --------------- QUICK SMOKE TEST (no training) ---------------
def quick_ctc_smoketest(model, data_loader, criterion, device, blank_idx=0, num_batches=1, decode_print=True):
    """
    Runs a few batches through the model and computes CTC loss.
    Prints shapes and a greedy decode sample to verify the pipeline.
    """
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            targets = batch['labels'].to(device)
            input_lengths = batch.get('input lengths').to(device)
            target_lengths = batch.get('target lengths').to(device)

            # logits = model(audio, video, mode="av")           # (B, T, C)
            logits,attn = model(audio_tensor=audio, visual_tensor=video, mode="av",
                            lengths_audio=input_lengths, lengths_video=input_lengths, attn_mask=None)
            print(f"[Batch {i}] logits: {tuple(logits.shape)}, "
                  f"targets: {tuple(targets.shape)}, "
                  f"in_len: {tuple(input_lengths.shape)}, "
                  f"tgt_len: {tuple(target_lengths.shape)}")

            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, C)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total += loss.item()

            if decode_print:
                decoded = _ctc_greedy_decode(logits, blank_idx=blank_idx)
                # Pretty print the first item
                idx_to_phoneme = batch.get('idx_to_phoneme', None)
                d0 = decoded[0].tolist() if len(decoded) > 0 else []
                if idx_to_phoneme is not None:
                    inv = idx_to_phoneme
                    printable = [inv.get(int(x), f"<{x}>") for x in d0]
                else:
                    printable = d0
                print(f"  Greedy decode (first item): {printable}")

            if i >= num_batches:
                break

    avg = total / max(1, min(num_batches, len(data_loader)))
    print(f"[SmokeTest] Avg CTC loss over {min(num_batches, len(data_loader))} batch(es): {avg:.4f}")
    return avg, attn

device = torch.device("cpu")
model = CrossAV(phoneme_vocab_size=43,
        a_drop_modality_prob=0.0,
        v_drop_modality_prob=0.0,
        return_attn=True,
        fusion_mask_type= "causal_band",    
        fusion_band= 5,              
        use_q_proj = True,
        use_k_proj= True,
        use_v_proj = True,).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
blank_idx = phoneme_to_idx['<BLANK>']  # should be 0
criterion = torch.nn.CTCLoss(blank=blank_idx, reduction = "mean", zero_infinity=True)
avg,attn=quick_ctc_smoketest(model, train_loader, criterion, device, blank_idx=0, num_batches=5, decode_print=True)





# --------- Setup logging ---------

# --- Training logger ---
logging.captureWarnings(True)

train_handler = logging.FileHandler("Aug26_training_warnings.log", mode="w")
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
val_handler = logging.FileHandler("Aug26_validation_warnings.log", mode="w")
val_handler.setLevel(logging.WARNING)
val_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

val_logger = logging.getLogger("val_logger")
val_logger.setLevel(logging.INFO)
val_logger.propagate = False  
val_logger.addHandler(val_handler)

# val_stream = logging.StreamHandler()
# val_stream.setFormatter(val_formatter)
# val_logger.addHandler(val_stream)


# ---------- Training loop ----------
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
        

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), 1):
            optimizer.zero_grad(set_to_none=True)
            # collate may return None if all items were filtered
            if batch is None:
                train_logger.warning(f"Empty batch at step {step},epoch {epoch}, skipping...")
                continue

            audio = batch['audio'].to(device)              # (B, T_a_max, F)
            video = batch['video'].to(device)              # (B, T_v_max, H, W)
            targets = batch['labels'].to(device)           # (sum_target,)
            in_lens = batch.get('input lengths').to(device).long()     # (B,)
            tgt_lens = batch.get('target lengths').to(device).long() # (B,)

            if epoch == 1 and step <= 5:
                model.audio_encoder.eval()
                model.visual_encoder.eval()
            else:
                model.audio_encoder.train()
                model.visual_encoder.train()
            
            with torch.amp.autocast("cuda",enabled=(use_amp and torch.cuda.is_available())):
                if model.return_attn:
                    logits, _ = model(
                        audio_tensor=audio, visual_tensor=video, mode="av",
                        lengths_audio=in_lens, lengths_video=in_lens, attn_mask=None
                    )   # (B, T, C)
                else:
                    logits = model(
                        audio_tensor=audio, visual_tensor=video, mode="av",
                        lengths_audio=in_lens, lengths_video=in_lens, attn_mask=None
                    )

                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)   # (T, B, C)

                B, T, C = logits.shape
                bad_targets = (targets < 0) | (targets >= C)
                if bad_targets.any():
                    idxs = bad_targets.nonzero(as_tuple=False).squeeze(-1)[:10]
                    train_logger.error(f"Target id out of range (>=C). idx={idxs.tolist()} vals={targets[idxs].tolist()} C={C}")
                    return

                if targets.numel() != int(tgt_lens.sum().item()):
                    train_logger.error(f"targets.numel()={targets.numel()} != sum(target_lengths)={int(tgt_lens.sum().item())}")
                    return

                impossible = (2*tgt_lens - 1) > in_lens     # feasibility check
                if impossible.any():
                    ids = impossible.nonzero(as_tuple=False).squeeze(-1).tolist()
                    train_logger.warning(f"CTC impossible items {ids} (2*L_tgt-1 > L_in); zero_infinity will zero these.")

                if not torch.isfinite(logits).all():
                    train_logger.error(f"NaN/Inf in logits at epoch {epoch}, step {step}. "
                                    f"min={float(np.nanmin(logits.detach().numpy()))} max={float(np.nanmax(logits.detach().numpy()))}")
                    return

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

        # Step LR scheduler
        scheduler.step(val_running_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"==> Epoch {epoch}: Train {train_running_loss:.4f} | Val {val_running_loss:.4f} | LR {current_lr:.6f}")

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
device = torch.device("cpu")

model = CrossAV(
    phoneme_vocab_size=43,
    a_drop_modality_prob=0.2,
    v_drop_modality_prob=0.1,
    return_attn=True,                # keep attention for optional analysis
    fusion_mask_type="causal_band",  # your choice
    fusion_band=6,
    use_q_proj=True,
    use_k_proj=True,
    use_v_proj=True,
    tcn_hidden=128, 
    tcn_layers=2, 
    tcn_kernel=3, 
    tcn_dropout=0.2, 
    tcn_dilation_base=2,
    use_mfcc_aug=True,
    use_frame_drop=True,
    p_video_frame_drop=0.04,
    frame_drop_mode='zero',
    time_mask_p=0.06,
    freq_mask_p=0.1,
    n_masks=1
).to(device)

# ---- (4) Criterion & Optimizer ----
# NOTE: blank index must match label space (0 is common)
criterion = CTCLoss(blank=0, reduction="mean", zero_infinity=True)

# A good starting optimizer for this kind of model
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",       # we want to minimize validation loss
    factor=0.5,       # LR *= 0.5 when triggered
    patience=2,       # wait 3 epochs of no improvement
    min_lr=1e-6,      # floor
    verbose=True      # print updates
)


# ---- (5) Checkpoint dir ----
ckpt_dir = "LRS_processing/LRS2/checkpoints/Aug26_test"
os.makedirs(ckpt_dir, exist_ok=True)

# ---- (6) Train! ----

summary = train_ctc(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    epochs=100,
    patience=6,
    grad_clip=1.0,
    log_interval=10,
    use_amp=False,         # automatically diable on CPU               
    ckpt_dir=ckpt_dir,
    log_json="Aug26_train_log.json",
)

print("Training summary:", summary)





























from torch.utils.data import DataLoader, Subset
import torch, random

# 1) Reuse the dataset behind your current loader
base_dataset   = train_loader.dataset
base_collate   = train_loader.collate_fn   # keep your existing collate
base_workers   = 0                         # safer on Mac
base_batchsize = 3                         # e.g., 3 items

# 2) Pick a tiny, deterministic subset (first K or random with a fixed seed)
K = 3
tiny_indices = list(range(K))              # or random.sample(range(len(base_dataset)), K)

tiny_train_ds = Subset(base_dataset, tiny_indices)
tiny_val_ds   = Subset(base_dataset, tiny_indices)  # for overfit test, you can reuse the same

tiny_train_loader = DataLoader(
    tiny_train_ds,
    batch_size=base_batchsize,
    shuffle=False,
    num_workers=base_workers,
    collate_fn=base_collate,
    drop_last=False,
)

tiny_val_loader = tiny_train_loader



summary = train_ctc(
    model=model,
    train_loader=tiny_train_loader,
    val_loader=tiny_val_loader,  # reuse for convenience
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    epochs=100,          # give it time to overfit
    patience=5,        # don’t early-stop the overfit test
    grad_clip=None,     # often disable for the overfit sanity check
    log_interval=1,
    use_amp=False,      # CPU/Mac
    ckpt_dir=ckpt_dir,
    log_json="overfit_log.json",
)
print("Overfit summary:", summary)

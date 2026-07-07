import torch
import random
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import random_split
from LRS_processing.build_dataloader import AVSRDataset,avsr_collate_fn
from LRS_processing.build_model import CrossAV
from LRS_processing.training_frameCE import train


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

device = get_device()
set_seed(seed = 20250811)

# run data set and split to train and validation sets 
dataset = AVSRDataset(
    audio_dir="LRS_processing/LRS2/LRS_main_audio",
    video_dir="LRS_processing/LRS2/LRS_main_mouth_roi",
    label_json_path="LRS_processing/LRS2/LRS_main_labelled_data/main_framewise_labels_cmu.json"
)

total_len = len(dataset)
train_len = int(0.6 * total_len)
val_len = (total_len - train_len)//2

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, val_len+1])

train_loader = DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True,
    num_workers=0,  # or 0 ?
    collate_fn=avsr_collate_fn
    )
val_loader = DataLoader(
    val_dataset, 
    batch_size=4, 
    shuffle=False,
    num_workers=0,  # or 0 ?
    collate_fn=avsr_collate_fn
    )
test_loader = DataLoader(
    test_dataset, 
    batch_size=4, 
    shuffle=False,
    num_workers=0,  # or 0 ?
    collate_fn=avsr_collate_fn
    )


model = CrossAV().to(device) 
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = CrossEntropyLoss(ignore_index=-100)  # mask padding labels

# train(model, train_loader, val_loader, optimizer, criterion, device, epochs=30, patience=5)




def test_run(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_frames = 0
    correct = 0

    with torch.no_grad():
        for batch in dataloader:
            audio_inputs = batch['audio'].to(device)      # (B, T, F)
            visual_inputs = batch['video'].to(device)     # (B, T, H, W)
            labels = batch['labels'].to(device)           # (B, T)

            diff = visual_inputs.size(1) - labels.size(1)

            if diff == -1:
                # copy and add the last frame (open cv read video by dropping the last a few ms if not perfectly aligned to frame rate. this will add it back)
                last_frame = visual_inputs[:, -1:].clone()   
                visual_inputs = torch.cat([visual_inputs, last_frame], dim=1)

            elif diff != 0:
                raise ValueError(f"Unexpected frame/label mismatch: diff={diff}")

            outputs = model(audio_inputs, visual_inputs)  # (B, T, num_classes)

            # Flatten for CE
            logits = outputs.view(-1, outputs.size(-1))   # (B*T, C)
            targets = labels.view(-1)                     # (B*T,)

            loss = criterion(logits, targets)
            total_loss += loss.item()

            # For quick sanity-check accuracy
            preds = logits.argmax(dim=-1)                 # (B*T,)
            mask = targets != -100                        # ignore padding
            correct += (preds[mask] == targets[mask]).sum().item()
            total_frames += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_frames if total_frames > 0 else 0
    print(f"Test loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")


test_run(model, train_loader, criterion, device)







## TESTING ##


with torch.no_grad():
        for batch in train_loader:
            audio_inputs = batch['audio']      # (B, T, F)
            visual_inputs = batch['video']     # (B, T, H, W)
            labels = batch['labels']          # (B, T)



import os
video_path = "LRS_processing/LRS2/LRS_main_mouth_roi/6098368658967223393_00001_ROI.mp4"
print(os.path.exists(video_path), os.path.getsize(video_path))


first_batch = next(iter(train_loader))

# Inspect what's inside
audio_inputs = first_batch['audio'].to(device)
visual_inputs = first_batch['video'].to(device)
labels = first_batch['labels'].to(device)

with torch.no_grad():

    diff = visual_inputs.size(1) - labels.size(1)

    if diff == -1:
        # copy and add the last frame (open cv read video by dropping the last a few ms if not perfectly aligned to frame rate. this will add it back)
        last_frame = visual_inputs[:, -1:].clone()   
        visual_inputs = torch.cat([visual_inputs, last_frame], dim=1)

    elif diff != 0:
        raise ValueError(f"Unexpected frame/label mismatch: diff={diff}")

    outputs = model(audio_inputs, visual_inputs)

    logits = outputs.view(-1, outputs.size(-1))   # (B*T, C)
    targets = labels.view(-1)                     # (B*T,)

    loss = criterion(logits, targets)


import cv2

first_batch['utt_ids'][1]

cap = cv2.VideoCapture(video_path)
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
cap.release()
print("Frames loaded:", count)

len(first_batch['labels'][1])
first_batch['video'][1].size()
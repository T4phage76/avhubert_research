import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import log_softmax
import copy

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=30, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            audio_inputs = batch['audio'].to(device)
            visual_inputs = batch['video'].to(device)
            targets = batch['labels'].to(device)

            outputs = model(audio_inputs, visual_inputs)  # (B, T, num_classes)
            outputs = outputs.view(-1, outputs.size(-1))  # (B*T, num_classes)
            targets = targets.view(-1)                    # (B*T,)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                audio_inputs = batch['audio'].to(device)
                visual_inputs = batch['video'].to(device)
                targets = batch['labels'].to(device)

                outputs = model(audio_inputs, visual_inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)

                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ↳ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    model.load_state_dict(best_model)



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

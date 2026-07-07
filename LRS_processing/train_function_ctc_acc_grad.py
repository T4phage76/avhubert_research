def train_ctc_grad_acc(
    # training data
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion, 
    device,
    # training parameters
    epochs=20, 
    patience=5, 
    grad_clip=1.0, 
    log_interval=40, 
    use_amp=True,
    accum_steps=3,  # [MODIFIED] gradient accumulation factor (micro-batches per optimizer step)
    # checkpointing:
    ckpt_dir="LRS_processing/LRS2/checkpoints/Cross_attn_test", 
    log_json="train_log.json"
):
    _safe_mkdir(ckpt_dir)

    best_val = float('inf')
    best_epoch = -1
    no_improve = 0

    log_history = []

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and torch.cuda.is_available()))
    accum_steps = max(1, int(accum_steps))  # [MODIFIED] guard

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        steps = 0

        optimizer.zero_grad(set_to_none=True)  # [MODIFIED] start epoch with cleared grads

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), 1):
            # collate may return None if all items were filtered
            if batch is None:
                train_logger.warning(f"Empty batch at step {step},epoch {epoch}, skipping...")
                continue

            audio = batch['audio'].to(device)              # (B, T_a_max, F)
            video = batch['video'].to(device)              # (B, T_v_max, H, W)
            targets = batch['labels'].to(device)           # (sum_target,)
            in_lens = batch.get('input lengths').to(device).long()     # (B,)
            tgt_lens = batch.get('target lengths').to(device).long()   # (B,)

            if epoch == 1 and step <= 5:
                model.audio_encoder.eval()
                model.visual_encoder.eval()
            else:
                model.audio_encoder.train()
                model.visual_encoder.train()
            
            with torch.amp.autocast("cuda", enabled=(use_amp and torch.cuda.is_available())):
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
                                       f"min={float(np.nanmin(logits.detach().cpu().numpy()))} max={float(np.nanmax(logits.detach().cpu().numpy()))}")
                    return

                train_batch_loss = criterion(log_probs, targets, in_lens, tgt_lens)

                # non-finite guard (train)
                if not torch.isfinite(train_batch_loss):
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

            # -------------- Gradient Accumulation -------------- #
            loss_for_backward = train_batch_loss / accum_steps  # [MODIFIED] scale loss to keep gradient magnitude
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()      # [MODIFIED]
            else:
                loss_for_backward.backward()                    # [MODIFIED]

            do_step = (step % accum_steps == 0)                 # [MODIFIED]
            if do_step:                                         # [MODIFIED]
                if grad_clip is not None:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)              # [MODIFIED]
                    clip_grad_norm_(model.parameters(), grad_clip)

                if scaler.is_enabled():
                    scaler.step(optimizer)                      # [MODIFIED]
                    scaler.update()                             # [MODIFIED]
                else:
                    optimizer.step()                            # [MODIFIED]

                optimizer.zero_grad(set_to_none=True)           # [MODIFIED]
            # --------------------------------------------------- #

            train_loss_accum += float(train_batch_loss.item())
            steps += 1
            train_running_loss = train_loss_accum / max(1, steps)

            if step % log_interval == 0:
                print(f"[Epoch {epoch} | Step {step}/{len(train_loader)}] "
                      f"Train CTC loss: {train_running_loss:.4f} (accum_steps={accum_steps})")  # [MODIFIED]
                
            log_history.append({
                "phase": "train",
                "epoch": epoch, 
                "step": step, 
                "running_loss": train_running_loss, 
                "batch_loss": float(train_batch_loss.item()),
                "accum_steps": accum_steps,  # [MODIFIED]
            })

        # ---- flush leftover grads if the epoch ends mid-accumulation ----
        leftover = (steps % accum_steps)                         # [MODIFIED]
        if leftover != 0:                                        # [MODIFIED]
            if grad_clip is not None:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        # ----------------------------------------------------------------

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

        # Step LR scheduler (kept as-is)
        scheduler.step(val_running_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"==> Epoch {epoch}: Train {train_running_loss:.4f} | Val {val_running_loss:.4f} | LR {current_lr:.6f} (accum_steps={accum_steps})")  # [MODIFIED]

        log_history.append({
            "phase": "validation",
            "epoch": epoch,
            "train_loss": float(train_running_loss),
            "val_loss": float(val_running_loss),
            "accum_steps": accum_steps,  # [MODIFIED]
        })

        # --------- Save checkpoints ---------
        last_path = _ckpt_name(ckpt_dir, epoch)
        _save_checkpoint(
            last_path, model, optimizer, scaler, epoch, val_running_loss,
            extra={"train_loss": float(train_running_loss), "accum_steps": accum_steps}  # [MODIFIED]
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

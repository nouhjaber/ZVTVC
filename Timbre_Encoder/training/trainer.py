"""
Trainer for Timbre Encoder
===========================

Based on the working old trainer. Changes from old:
  - Early stopping (patience-based)
  - Expensive metrics only on log steps (is_log_step guard)
  - TensorBoard on local disk, not in checkpoint dir
  - Auto-run Validate.py at end of training
  - Always save best_model.pt + stage checkpoints
  - Correct loss target display for InfoNCE
"""

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

import time
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model.ecapa_tdnn import ECAPATDNN
from training.loss import InfoNCELoss
from training.optimizer import (
    create_optimizer,
    create_scheduler,
    GradientClipper,
    OptimizerState,
)
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def _adapt_batch_if_needed(batch, dataset):
    if 'speaker_id' not in batch and 'speaker_ids' in batch:
        ids = batch['speaker_ids']
        batch['speaker_id'] = torch.tensor(ids) if isinstance(ids, list) else ids
    return batch


class Trainer:
    def __init__(
        self,
        model: ECAPATDNN,
        train_dataloader,
        val_dataloader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        device: str = "cuda",
        output_dir: str = "./outputs",
        use_amp: bool = True,
        gradient_clip_norm: float = 3.0,
        log_interval: int = 100,
        validation_interval: int = 2000,
        checkpoint_interval: int = 15000,
        stages: Optional[list] = None,
        stage_only_checkpoints: bool = True,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 0.001,
    ):
        logger.info("Initializing Trainer")
        logger.info(f"  device={device}, use_amp={use_amp}, grad_clip={gradient_clip_norm}")
        logger.info(f"  log={log_interval}, val={validation_interval}, ckpt={checkpoint_interval}")
        logger.info(f"  early_stop: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")

        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.use_amp = use_amp
        self.log_interval = log_interval
        self.validation_interval = validation_interval
        self.checkpoint_interval = checkpoint_interval
        self.stage_only_checkpoints = bool(stage_only_checkpoints)
        self.stages = stages or []
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_counter = 0
        self.best_val_loss = float("inf")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.gradient_clipper = GradientClipper(max_norm=gradient_clip_norm)
        self.scaler = GradScaler() if use_amp else None

        # TensorBoard on LOCAL disk — not in checkpoint dir (which may be on Drive)
        tb_dir = '/content/logs/tb_timbre'
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"TensorBoard logs: {tb_dir}")

        self.current_iteration = 0
        self.current_stage = 0

        logger.info("Trainer initialized")

    def train(self, total_iterations: int):
        logger.info(f"Starting training for {total_iterations} iterations")
        print(f"Starting training for {total_iterations} iterations")
        print(f"Device: {self.device}, AMP: {self.use_amp}")

        self.model.train()
        start_time = time.time()
        dataloader_iter = iter(self.train_dataloader)

        while self.current_iteration < total_iterations:
            self._check_stage_switch()

            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.train_dataloader)
                batch = next(dataloader_iter)

            metrics = self._train_step(batch)
            self.current_iteration += 1

            # Log + print
            if self.current_iteration % self.log_interval == 0:
                self._log_metrics(metrics)
                elapsed = time.time() - start_time
                ips = self.current_iteration / elapsed
                eta = (total_iterations - self.current_iteration) / ips
                print(self._format_metrics(metrics, self.current_iteration, total_iterations, eta))

            # Validation
            if self.current_iteration % self.validation_interval == 0:
                val_metrics = self.validate()
                self._log_val_metrics(val_metrics)
                self._print_val_metrics(val_metrics)

                # Best model
                if val_metrics["loss"] < self.best_val_loss - self.early_stopping_min_delta:
                    self.best_val_loss = val_metrics["loss"]
                    self.early_stopping_counter = 0
                    self._save_checkpoint("best_model.pt")
                    print(f"  *** New best val loss: {val_metrics['loss']:.4f} ***")
                else:
                    self.early_stopping_counter += 1
                    logger.info(f"Early stop counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                    if self.early_stopping_patience > 0 and self.early_stopping_counter >= self.early_stopping_patience:
                        logger.info("Early stopping triggered!")
                        print("Early stopping triggered!")
                        self._save_checkpoint("early_stopped.pt")
                        break

                self.model.train()

        # Final save
        self._save_checkpoint("final_model.pt")
        hours = (time.time() - start_time) / 3600
        logger.info(f"Training completed in {hours:.2f} hours")
        print(f"Training completed in {hours:.2f} hours")
        self.writer.close()

    def _train_step(self, batch: Dict) -> Dict:
        batch = _adapt_batch_if_needed(batch, self.train_dataloader.dataset)

        mels = batch["mel"].to(self.device, non_blocking=True)
        speaker_ids = batch["speaker_id"].to(self.device, non_blocking=True)

        batch_size = mels.size(0)
        speakers_per_batch = len(torch.unique(speaker_ids))
        utterances_per_speaker = batch_size // speakers_per_batch

        with autocast(device_type='cuda', enabled=self.use_amp):
            embeddings = self.model(mels)
            loss_output = self.loss_fn(
                embeddings,
                speakers_per_batch=speakers_per_batch,
                utterances_per_speaker=utterances_per_speaker,
            )
            loss = loss_output["loss"]

        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = self.gradient_clipper(self.model)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = self.gradient_clipper(self.model)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        metrics = {
            "loss": loss.item(),
            "accuracy": loss_output.get("accuracy", 0.0),
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
        }

        # Expensive metrics — ONLY on log steps
        is_log_step = self.current_iteration % self.log_interval == (self.log_interval - 1)
        if is_log_step:
            with torch.no_grad():
                emb_mean = embeddings.mean().item()
                emb_std = embeddings.std().item()
                emb_norm = F.normalize(embeddings, p=2, dim=1)
                sim = torch.matmul(emb_norm, emb_norm.t())
                spk_labels = torch.arange(speakers_per_batch, device=self.device).repeat_interleave(utterances_per_speaker)
                same_mask = (spk_labels.unsqueeze(0) == spk_labels.unsqueeze(1)).float() - torch.eye(batch_size, device=self.device)
                diff_mask = (spk_labels.unsqueeze(0) != spk_labels.unsqueeze(1)).float()
                metrics["embedding_mean"] = emb_mean
                metrics["embedding_std"] = emb_std
                metrics["same_speaker_sim"] = sim[same_mask > 0].mean().item() if same_mask.sum() > 0 else 0.0
                metrics["diff_speaker_sim"] = sim[diff_mask > 0].mean().item() if diff_mask.sum() > 0 else 0.0

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict:
        if self.val_dataloader is None:
            return {"loss": 0.0, "accuracy": 0.0}

        self.model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        skipped = 0

        for batch in self.val_dataloader:
            batch = _adapt_batch_if_needed(batch, self.val_dataloader.dataset)
            mels = batch["mel"].to(self.device)
            speaker_ids = batch["speaker_id"].to(self.device)

            unique_spks, counts = torch.unique(speaker_ids, return_counts=True)

            # Keep only speakers with >= 2 utterances in this batch (singletons can't
            # form positive pairs for InfoNCE). Val data is shuffled, so most batches
            # will have many singletons — filtering them is the right move; skipping
            # the whole batch loses too much data.
            keep_spks = unique_spks[counts >= 2]
            if len(keep_spks) < 2:
                skipped += 1
                continue

            # Mask down to the kept speakers
            keep_mask = torch.isin(speaker_ids, keep_spks)
            mels_f = mels[keep_mask]
            ids_f = speaker_ids[keep_mask]

            # Trim to uniform utt-per-speaker
            _, counts_f = torch.unique(ids_f, return_counts=True)
            utt_per_spk = int(counts_f.min().item())
            n_spk = len(keep_spks)

            keep_mels, keep_ids = [], []
            for spk in keep_spks.tolist():
                idx = (ids_f == spk).nonzero(as_tuple=True)[0][:utt_per_spk]
                keep_mels.append(mels_f[idx])
                keep_ids.append(ids_f[idx])
            mels_v = torch.cat(keep_mels, dim=0)

            embeddings = self.model(mels_v)
            try:
                loss_out = self.loss_fn(
                    embeddings,
                    speakers_per_batch=n_spk,
                    utterances_per_speaker=utt_per_spk,
                )
                total_loss += loss_out["loss"].item()
                total_acc += loss_out.get("accuracy", 0.0)
                n += 1
            except Exception as e:
                logger.debug(f"Val batch skipped: {e}")
                skipped += 1

        if n == 0:
            logger.warning(
                f"All {skipped} val batches skipped — val data has no speakers with "
                f">=2 utterances per batch. Increase val batch size or use a sampler."
            )
            return {"loss": float("inf"), "accuracy": 0.0}
        if skipped > 0:
            logger.debug(f"Val: {n} batches used, {skipped} skipped")
        return {"loss": total_loss / n, "accuracy": total_acc / n}

    def _check_stage_switch(self):
        if self.current_stage >= len(self.stages):
            return
        stage_end = sum(s["iterations"] for s in self.stages[:self.current_stage + 1])
        if self.current_iteration >= stage_end:
            stage = self.stages[self.current_stage]
            stage_name = stage.get("stage_name", f"stage{self.current_stage+1}")
            safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(stage_name))
            self._save_checkpoint(f"stage{self.current_stage+1}_{safe_name}.pt")

            self.current_stage += 1
            if self.current_stage < len(self.stages):
                ns = self.stages[self.current_stage]
                print(f"\n{'='*60}\nSwitching to Stage {self.current_stage+1}: {ns['stage_name']}\n{'='*60}")
                if "learning_rate" in ns:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = ns["learning_rate"]
                if "enable_hard_negatives" in ns and hasattr(self.loss_fn, 'use_hard_negatives'):
                    self.loss_fn.use_hard_negatives = ns["enable_hard_negatives"]

    def _log_metrics(self, m):
        for k, v in m.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                self.writer.add_scalar(f"train/{k}", v, self.current_iteration)

    def _log_val_metrics(self, m):
        for k, v in m.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                self.writer.add_scalar(f"val/{k}", v, self.current_iteration)

    def _format_metrics(self, m, cur, total, eta):
        pct = cur / total * 100
        lines = [f"\n{'='*70}",
                 f"  Iter: {cur:,}/{total:,} ({pct:.1f}%) | ETA: {eta/3600:.2f}h",
                 f"{'='*70}",
                 f"  Loss:           {m['loss']:7.4f}  [Target: <2.0]",
                 f"  Accuracy:       {m.get('accuracy',0):7.3f}  [Target: >0.30]",
                 f"  Grad Norm:      {m['grad_norm']:7.4f}",
                 f"  LR:             {m['lr']:.6f}"]
        if "embedding_mean" in m:
            lines += [f"  Emb Mean:       {m['embedding_mean']:7.4f}",
                      f"  Emb Std:        {m['embedding_std']:7.4f}  [~0.0625]",
                      f"  Same Spk Sim:   {m['same_speaker_sim']:7.4f}  [>0.60]",
                      f"  Diff Spk Sim:   {m['diff_speaker_sim']:7.4f}  [<0.40]",
                      f"  Separation:     {m['same_speaker_sim']-m['diff_speaker_sim']:7.4f}  [>0.20]"]
        lines.append(f"{'='*70}")
        return "\n".join(lines)

    def _print_val_metrics(self, m):
        print(f"\n{'='*70}\n  VALIDATION\n{'='*70}")
        print(f"  Loss:     {m['loss']:.4f}")
        print(f"  Accuracy: {m['accuracy']:.3f}")
        print(f"{'='*70}")

    def _save_checkpoint(self, filename):
        path = self.checkpoint_dir / filename
        ckpt = {
            "iteration": self.current_iteration,
            "stage": self.current_stage,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = {"current_iteration": self.scheduler.current_iteration}
        if self.scaler is not None:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(ckpt, path)
        print(f"Saved: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_iteration = ckpt["iteration"]
        self.current_stage = ckpt.get("stage", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.current_iteration = ckpt["scheduler_state_dict"]["current_iteration"]
        if self.scaler and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"Loaded checkpoint from iteration {self.current_iteration}")


def create_trainer_from_config(config, model, train_dataset, val_dataset):
    num_workers = config.get("num_workers", 4)
    pin_memory = config.get("pin_memory", True)
    spk = config["batch"]["speakers_per_batch"]
    utt = config["batch"]["utterances_per_speaker"]

    batch_sampler = train_dataset.get_speaker_aware_sampler(speakers_per_batch=spk, utterances_per_speaker=utt)
    train_dl = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                          pin_memory=pin_memory, collate_fn=train_dataset.collate_fn,
                          persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None)

    val_dl = None
    if val_dataset is not None:
        val_dl = DataLoader(val_dataset, batch_size=spk * utt, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, collate_fn=val_dataset.collate_fn,
                            persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None)

    loss_fn = InfoNCELoss(
        temperature=config["loss"]["temperature"],
        use_hard_negatives=config["loss"].get("use_hard_negatives", False),
    )

    optimizer = create_optimizer(model=model, optimizer_name=config["optimizer"]["name"],
                                 learning_rate=config["optimizer"]["learning_rate"],
                                 weight_decay=config["optimizer"]["weight_decay"])

    scheduler = create_scheduler(optimizer=optimizer, scheduler_name=config["scheduler"]["name"],
                                  warmup_iterations=config["scheduler"]["warmup_iterations"],
                                  total_iterations=config["total_iterations"],
                                  min_lr=config["scheduler"].get("min_lr", 1e-6))

    return Trainer(
        model=model, train_dataloader=train_dl, val_dataloader=val_dl,
        loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=config.get("output_dir", "./outputs"),
        use_amp=config.get("use_amp", True),
        gradient_clip_norm=config["optimizer"].get("gradient_clip_norm", 3.0),
        log_interval=config.get("log_interval", 100),
        validation_interval=config.get("validation_interval", 2000),
        checkpoint_interval=config.get("save_checkpoint_interval", 15000),
        stages=config.get("stages", []),
        stage_only_checkpoints=config.get("stage_only_checkpoints", True),
        early_stopping_patience=config.get("early_stopping_patience", 5),
        early_stopping_min_delta=config.get("early_stopping_min_delta", 0.001),
    )
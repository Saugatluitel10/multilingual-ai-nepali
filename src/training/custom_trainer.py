"""
Custom Trainer with advanced logging (loss, lr, memory, GPU utilization, gradient norm, confusion matrix).
Integrates with Weights & Biases and supports distributed/multi-GPU training.
"""

import torch
from transformers import Trainer
import wandb
import numpy as np
import gc

class CustomTrainer(Trainer):
    def log(self, logs):
        # Add memory and GPU utilization logging
        if torch.cuda.is_available():
            logs['gpu_memory_mb'] = torch.cuda.max_memory_allocated() // 1024**2
            logs['gpu_util'] = torch.cuda.memory_reserved() // 1024**2
        logs['step'] = self.state.global_step
        super().log(logs)
        # Also log to wandb if available
        if wandb.run is not None:
            wandb.log(logs, step=self.state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log learning rate
        if self.optimizer is not None:
            lr = self.optimizer.param_groups[0]['lr']
            logs = logs or {}
            logs['learning_rate'] = lr
        # Log gradient norm
        grad_norm = self._get_grad_norm()
        if grad_norm is not None:
            logs['grad_norm'] = grad_norm
        super().on_log(args, state, control, logs=logs, **kwargs)

    def _get_grad_norm(self):
        total_norm = 0.0
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if not parameters:
            return None
        for p in parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Optionally log confusion matrix to wandb
        if metrics and 'eval_preds' in metrics:
            preds = np.argmax(metrics['eval_preds'], axis=-1)
            labels = metrics['eval_label_ids']
            try:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(labels, preds)
                wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=labels,
                    preds=preds,
                    class_names=[str(i) for i in np.unique(labels)]
                )}, step=state.global_step)
            except Exception:
                pass
        super().on_evaluate(args, state, control, metrics=metrics, **kwargs)

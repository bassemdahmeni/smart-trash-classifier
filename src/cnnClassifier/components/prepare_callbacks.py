import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config
        self.writer = None
        self.ckpt_path = self.config.checkpoint_model_filepath

    def _create_tb_writer(self):
        """Create a TensorBoard writer with timestamped log dir"""
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        os.makedirs(tb_running_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_running_log_dir)
        print(f"[TensorBoard] Logging at: {tb_running_log_dir}")
        return self.writer

    def _save_checkpoint(self, model, optimizer, epoch, val_loss):
        """Save model checkpoint if it's the best so far"""
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, self.ckpt_path)
        print(f"[Checkpoint] Model saved at {self.ckpt_path}")

    def get_tb_writer(self):
        """Public method to get TensorBoard writer"""
        if self.writer is None:
            self._create_tb_writer()
        return self.writer

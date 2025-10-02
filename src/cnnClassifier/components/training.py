import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig

class TrainModel:
    def __init__(self, config: TrainingConfig, model: nn.Module,
                 train_loader, val_loader,callbacks=None, device="cuda"):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.callbacks = callbacks if callbacks else []
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load pretrained (updated) weights if available
        if self.config.pretrained_model_path and Path(self.config.pretrained_model_path).exists():
            print(f"📥 Loading pretrained weights from {self.config.pretrained_model_path}")
            self.model.load_state_dict(torch.load(self.config.pretrained_model_path, map_location=device))

        # Loss, optimizer, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train(self):
        print(f"\n🚀 Starting Training for {self.config.num_epochs} epochs")
        print(f"   - Learning rate: {self.config.learning_rate}")
        print(f"   - Weight decay: {self.config.weight_decay}")
        print(f"   - Patience: {self.config.patience}")
        print(f"   - Unfreezing last {self.config.num_layers_to_unfreeze} layers")

        best_acc = 0.0
        patience_counter = 0
         # 🔹 If TensorBoard is available
        tb_writer = None
        for cb in self.callbacks:
            if hasattr(cb, "get_tb_writer"):
                tb_writer = cb.get_tb_writer()

        for epoch in range(self.config.num_epochs):
            # 🔹 Training
            self.model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            train_acc = 100 * correct_train / total_train
            epoch_loss = running_loss / len(self.train_loader.dataset)
            

            # 🔹 Validation
            self.model.eval()
            correct_val, total_val = 0, 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, preds = torch.max(outputs, 1)
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)

            val_acc = 100 * correct_val / total_val

            print(f"📊 Epoch {epoch+1}/{self.config.num_epochs} "
                  f"| Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            # 🔹 TensorBoard logging
            if tb_writer:
                tb_writer.add_scalar("Loss/train", epoch_loss, epoch)
                tb_writer.add_scalar("Accuracy/train", train_acc, epoch)
                tb_writer.add_scalar("Accuracy/val", val_acc, epoch)
            # 🔹 Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                for cb in self.callbacks:
                    if hasattr(cb, "_save_checkpoint"):
                        cb._save_checkpoint(self.model, self.optimizer, epoch, epoch_loss)
                torch.save(self.model.state_dict(), self.config.trained_model_path)
                print(f"   💾 Best model updated @ {val_acc:.2f}% "
                      f"-> saved to {self.config.trained_model_path}")
                patience_counter = 0
            else:
                patience_counter += 1

            # 🔹 Early stopping
            if patience_counter >= self.config.patience:
                print("⏹️ Early stopping triggered")
                break

            self.scheduler.step()

        print(f"✅ Training completed! Best Val Accuracy: {best_acc:.2f}%")
        return best_acc

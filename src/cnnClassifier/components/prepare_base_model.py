import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
import os



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        print(f"[Init] PrepareBaseModel initialized with model: {self.config.model_name}")

    def get_base_model(self) -> nn.Module:
        """Load EfficientNet backbone based on config"""
        print("[Step 1] Loading base model...")
        if self.config.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=self.config.params_weights)
        elif self.config.model_name == "efficientnet_b1":
            model = models.efficientnet_b1(weights=self.config.params_weights)
        else:
            raise ValueError(f"Model {self.config.model_name} not supported")

        # Freeze layers if include_top is False
        if not self.config.params_include_top:
            print("[Step 2] Freezing base model layers...")
            for param in model.parameters():
                param.requires_grad = False

        # Replace classifier head
        print("[Step 3] Replacing classifier head...")
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, self.config.params_classes)
        )

        print("[Done] Base model created.")
        return model

    def save_model(self, model: nn.Module, path: Path):
        """Save model weights"""
        torch.save(model.state_dict(), path)
        print(f"[Saved] Model saved at: {path}")

    def prepare(self):
        """Prepare and save the base model"""
        print("[Pipeline] Preparing base model...")
        model = self.get_base_model()
        self.save_model(model, self.config.base_model_path)
        print("[Pipeline Done] Base model prepared and saved.")
        return model

    def update_base_model(self, num_layers_to_unfreeze: int = 50):
        """
        Load the base model, unfreeze last N layers + classifier for fine-tuning,
        then save it as updated base model.
        """
        print("[Pipeline] Updating base model...")
        model = self.get_base_model()
        model.load_state_dict(torch.load(self.config.base_model_path))
        print("[Step 1] Loaded base model weights.")

        # Always unfreeze classifier
        print("[Step 2] Unfreezing classifier...")
        for param in model.classifier.parameters():
            param.requires_grad = True

        # Unfreeze last few backbone layers
        total_layers = len(list(model.features))
        layers_to_unfreeze = min(num_layers_to_unfreeze, total_layers)
        print(f"[Step 3] Unfreezing last {layers_to_unfreeze} backbone layers...")

        for i, child in enumerate(reversed(list(model.features.children()))):
            if i < layers_to_unfreeze:
                for param in child.parameters():
                    param.requires_grad = True

        # Save updated model
        self.save_model(model, self.config.updated_base_model_path)
        print("[Pipeline Done] Updated base model ready for fine-tuning.")

        return model

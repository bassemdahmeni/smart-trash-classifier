import torch
import torch.nn as nn
from pathlib import Path
from cnnClassifier.utils.common import save_json
from cnnClassifier.entity.config_entity import EvaluationConfig

class Evaluation:
    def __init__(self, config: "EvaluationConfig", model: nn.Module, test_data, device="cuda"):
        
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"

        # Initialize model architecture
        self.model = model.to(self.device)

        # Load saved weights
        self.model.load_state_dict(
            torch.load(self.config.path_of_model, map_location=self.device)
        )
        self.model.eval()

        self.test_data = test_data
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self):
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_data:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = test_loss / total
        accuracy = correct / total * 100
        return {"loss": avg_loss, "accuracy": accuracy}

    def save_results(self, results: dict):
        """Save evaluation results as JSON"""
        save_json(Path("scores.json"), results)

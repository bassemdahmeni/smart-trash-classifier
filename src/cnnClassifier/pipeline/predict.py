import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

from cnnClassifier import logger
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline








class PredictionPipeline:
    def __init__(self, model_path: Path, class_names=None, device=None):
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        
        # Load model architecture (same as training)
        path_base_model = Path("artifacts/prepare_base_model/base_model_arch.pth")
        self.model = torch.load(path_base_model, map_location=self.device,weights_only=False)

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"PredictionPipeline initialized with model at {self.model_path}")

    def predict(self, image_path: str) -> dict:
        """Run prediction on a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class = torch.argmax(outputs[0]).item()
                confidence = probabilities[predicted_class].item()

            # Convert to dictionary
            all_probs = {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities.cpu().numpy())
            }

            result = {
                "predicted_class": self.class_names[predicted_class],
                "confidence": confidence,
                "all_probabilities": all_probs
            }

            logger.info(f"Prediction complete: {result['predicted_class']} ({result['confidence']:.2f})")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": f"Failed to process image: {str(e)}"}


if __name__ == "__main__":
    predictor = PredictionPipeline(model_path="artifacts/training/model.pth")
    result = predictor.predict("artifacts\data_ingestion\dataset-resized\glass\glass9.jpg")
    print(result)
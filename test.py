
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
import torch
from pathlib import Path




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# prepare_pipeline = PrepareBaseModelTrainingPipeline()
# updated_model, model_architecture = prepare_pipeline.main()
# torch.save(model_architecture, "artifacts/model_architecture.pth")
# print("✅ Model architecture saved successfully.")

path=Path("artifacts\\prepare_base_model\\base_model_arch.pth")
model = torch.load(path, map_location=device, weights_only=False)
print(model)
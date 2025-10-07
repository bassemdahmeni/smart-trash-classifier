🗑️ Smart Waste Classification System
An end-to-end deep learning application that classifies waste materials into different categories to facilitate proper recycling and disposal.
Built with PyTorch, Flask, and modern MLOps practices.

🚀 Features
Multi-class Waste Classification — Classifies images into 6 categories: cardboard, glass, metal, paper, plastic, and trash

End-to-End Pipeline — Complete ML workflow from data ingestion to deployment

Modern Architecture — Uses EfficientNet with fine-tuning for optimal performance

REST API — Flask-based web interface for easy integration

MLOps Ready — DVC for data versioning, GitHub Actions for CI/CD, Docker for containerization

🏗️ System Architecture
🔹 Pipeline Stages
Data Ingestion — Downloads and organizes the waste dataset

Base Model Preparation — Loads and configures EfficientNet backbone

Data Preprocessing — Applies transformations and creates data loaders

Model Training — Fine-tunes the model with early stopping and callbacks

Model Evaluation — Validates performance on the test set

Prediction API — Real-time classification via REST endpoints

🔹 Model Details
Backbone: EfficientNet-B0/B1 (pretrained weights)

Classifier Head: Custom dense layers with dropout for regularization

Fine-Tuning: Progressive unfreezing of backbone layers

Optimization: Adam optimizer with StepLR scheduler

Loss Function: Cross-Entropy with early stopping

🛠️ Installation & Setup
🔧 Prerequisites
Python 3.10

Conda / Miniconda

Git

DVC (for data versioning)

⚙️ Local Development
# Clone the repository
git clone <your-repo-url>
cd trash-classification-app

# Create and activate conda environment
conda create -n trash-classification python=3.10
conda activate trash-classification

# Install dependencies
pip install -r requirements.txt

# Setup DVC (if using remote storage)
dvc pull

🐳 Using Docker
# Build the image
docker build -t trash-classification-app .

# Run the container
docker run -p 8080:8080 trash-classification-app

📖 Usage
🧠 Training the Model
# Run the complete training pipeline
python main.py

# Or via API:
curl -X POST http://localhost:8080/train

train
🧩 Making Predictions
Via API:
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_encoded_image>"}'

from cnnClassifier.pipeline.predict import PredictionPipeline

predictor = PredictionPipeline()
result = predictor.predict("path/to/your/image.jpg")
print(result)

🌐 API Endpoints
Method	Endpoint	Description
GET	/	Home page with web interface
POST	/train	Trigger model training pipeline
POST	/predict	Classify waste image (expects base64)
🏭 MLOps Infrastructure
📦 Data Version Control (DVC)
Tracks datasets and model artifacts

Ensures reproducible pipelines

Supports remote storage integration

⚙️ CI/CD with GitHub Actions
Automated testing on push/pull requests

Docker image building and publishing

Deployment automation

🐳 Containerization
Fully Dockerized for consistent environments

Conda environment management

Easily deployable to cloud platforms

🌐 Deployment
💻 Local Deployment
python app.py
# Access at http://localhost:8080

☁️ Cloud Deployment
AWS: Use port 8080

Azure: Use port 80

Other: Adjust ports in app.py as needed

🔧 Configuration
Key configuration parameters are defined in config_entity.py:

Model architecture (EfficientNet B0/B1)

Learning rate & optimizer settings

Data augmentation parameters

Training hyperparameters

Path configurations

🤝 Contributing
Fork the repository

Create a new feature branch

Commit your changes

Push to your branch

Create a Pull Request 🚀
import kagglehub
import os
import zipfile
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            try:
                logger.info(f"Downloading dataset from Kaggle: {self.config.source_URL}")
                
                # Extract dataset owner and name from URL
                # Example URL: "https://www.kaggle.com/datasets/techsash/waste-classification-data"
                url_parts = self.config.source_URL.strip('/').split('/')
                dataset_owner = url_parts[-2]
                dataset_name = url_parts[-1]
                
                logger.info(f"Downloading dataset: {dataset_owner}/{dataset_name}")
                
                # Download dataset using kagglehub
                path = kagglehub.dataset_download(f"{dataset_owner}/{dataset_name}")
                
                # The downloaded file might be a zip file or directory
                # If it's a directory, we need to zip it or handle accordingly
                if os.path.isdir(path):
                    # Create a zip file from the downloaded directory
                    with zipfile.ZipFile(self.config.local_data_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for root, dirs, files in os.walk(path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, path)
                                zipf.write(file_path, arcname)
                    logger.info(f"Downloaded dataset zipped to: {self.config.local_data_file}")
                else:
                    # If it's already a file, move/copy it to our desired location
                    import shutil
                    shutil.move(path, self.config.local_data_file)
                    logger.info(f"Downloaded file moved to: {self.config.local_data_file}")
                
                logger.info(f"Dataset downloaded successfully!")
                
            except Exception as e:
                logger.error(f"Error downloading dataset from Kaggle: {e}")
                raise e
        else:
            logger.info(f"File already exists ")  


    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        logger.info(f"Extracting zip file: {self.config.local_data_file}")
        logger.info(f"Extracting to: {unzip_path}")
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        
        logger.info(f"Extraction completed to: {unzip_path}")
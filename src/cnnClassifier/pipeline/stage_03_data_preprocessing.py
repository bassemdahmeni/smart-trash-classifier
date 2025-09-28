from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_preprocessing import DataPreprocessing
from cnnClassifier import logger

STAGE_NAME = "Data Preprocessing stage"


class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        preprocessing = DataPreprocessing(config=data_preprocessing_config)
        
        train_loader, val_loader, test_loader = preprocessing.get_dataloaders()

        # Print some debug info
        logger.info(f"Train loader batches: {len(train_loader)}")
        logger.info(f"Validation loader batches: {len(val_loader)}")
        logger.info(f"Test loader batches: {len(test_loader)}")

        # Optional sanity check: get a batch
        for images, labels in train_loader:
            logger.info(f"Sample batch - images shape: {images.shape}, labels shape: {labels.shape}")
            break

        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

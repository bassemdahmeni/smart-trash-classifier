from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.training import TrainModel
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_data_preprocessing import DataPreprocessingTrainingPipeline
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier import logger
import torch







STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self,model_architecture=None, train_loader=None, val_loader=None):
            config = ConfigurationManager()
            training_config = config.get_training_config()
            prepare_callbacks_config = config.get_prepare_callback_config()
            prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
            
            model_architecture = torch.load(training_config.model_architecture_path,weights_only=False)
            # 3. Initialize Training component
            trainer = TrainModel(
                model=model_architecture,
                config=training_config,
                train_loader=train_loader,
                val_loader=val_loader,
                callbacks=[prepare_callbacks]
                
            )

            # 4. Train the model
            trainer.train()




if __name__ == '__main__':
    try:
       
        
        data_pipeline = DataPreprocessingTrainingPipeline()
        train_loader, val_loader, test_loader = data_pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main( train_loader=train_loader, val_loader=val_loader)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
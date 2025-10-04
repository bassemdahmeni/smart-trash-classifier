from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.evaluation import Evaluation
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_data_preprocessing import DataPreprocessingTrainingPipeline
from cnnClassifier import logger
import torch


STAGE_NAME = "Evaluation"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self, model_architecture=None, test_loader=None):
        # Load configuration
        config = ConfigurationManager()
        eval_config = config.get_validation_config()   # <-- you’ll add this in ConfigurationManager
        model_architecture = torch.load(eval_config.model_architecture_path,weights_only=False)
        # Initialize evaluator
        evaluator = Evaluation(
            config=eval_config,
            model=model_architecture,   # architecture (class) to build model
            test_data=test_loader
            
        )
        
        # Run evaluation
        results = evaluator.evaluate()
        logger.info(f"Evaluation Results: {results}")

        # Save results as JSON
        evaluator.save_results(results)
        


if __name__ == '__main__':
    try:
        

        # Get data loaders
        data_pipeline = DataPreprocessingTrainingPipeline()
        train_loader, val_loader, test_loader = data_pipeline.main()

        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main(test_loader=test_loader)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e

from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig
import os
from cnnClassifier.entity.config_entity import DataPreprocessingConfig
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.entity.config_entity import EvaluationConfig
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            model_name=self.params.MODEL_NAME,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath)
        )

        return prepare_callback_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config_root_data = self.config.data_ingestion
        config = DataPreprocessingConfig(
            data_dir=Path(config_root_data.unzip_dir,"dataset-resized"),
            batch_size=self.params.BATCH_SIZE,
            image_size=self.params.IMAGE_SIZE,
            val_split=self.params.VAL_SPLIT,
            test_split=self.params.TEST_SPLIT,
            shuffle=self.params.SHUFFLE_DATASET,
            random_seed=self.params.RANDOM_SEED,
            augmentation=self.params.AUGMENTATION
        )
        return config
    
    def get_training_config(self) -> TrainingConfig:
        training_cfg = self.config.training
        pretrained_model_path = self.config.prepare_base_model.updated_base_model_path
        create_directories([training_cfg.root_dir])
        return TrainingConfig(
            root_dir=training_cfg.root_dir,
            trained_model_path=training_cfg.trained_model_path,
            pretrained_model_path=pretrained_model_path,
            learning_rate=0.0001,          # you can push this to yaml later
            weight_decay=1e-4,
            num_epochs=5,
            patience=5,
            num_layers_to_unfreeze=30
        )
    
    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.pth",
            training_data="artifacts/data_ingestion/dataset-resized",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    model_name: str
    params_classes: int


@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path


@dataclass(frozen=True)
class DataPreprocessingConfig:
    data_dir: Path
    batch_size: int
    image_size: list
    val_split: float
    test_split: float
    shuffle: bool
    random_seed: int
    augmentation: bool

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: str
    trained_model_path: str
    learning_rate: float
    weight_decay: float
    num_epochs: int
    patience: int
    num_layers_to_unfreeze: int
    pretrained_model_path: str
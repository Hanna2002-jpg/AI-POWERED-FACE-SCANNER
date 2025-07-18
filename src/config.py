"""
Configuration Management System
==============================

Centralized configuration management with validation and environment support.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    architecture: str = "arcface"
    backbone: str = "resnet50"
    embedding_size: int = 512
    pretrained: bool = True
    dropout_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 20
    optimizer: str = "adam"
    scheduler: str = "steplr"
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Fine-tuning strategy
    freeze_backbone: bool = False
    unfreeze_after_epoch: int = 5
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_strength: float = 0.5


@dataclass
class DataConfig:
    """Data configuration parameters."""
    train_split: float = 0.7
    test_split: float = 0.3
    image_size: tuple = (224, 224)
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)
    
    # Quality assessment
    min_face_size: int = 50
    blur_threshold: float = 100.0
    illumination_threshold: float = 50.0


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    similarity_threshold: float = 0.5
    distance_metric: str = "cosine"
    
    # Threshold analysis
    threshold_range: tuple = (0.1, 0.9)
    threshold_steps: int = 9
    
    # Visualization
    plot_roc: bool = True
    plot_precision_recall: bool = True
    plot_similarity_distribution: bool = True


class Config:
    """Main configuration class with validation and environment support."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = Path(config_path)
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.evaluation = EvaluationConfig()
        
        # Paths
        self.data_path: Optional[str] = None
        self.output_dir: str = "outputs"
        self.checkpoint_dir: str = "outputs/checkpoints"
        self.logs_dir: str = "outputs/logs"
        self.reports_dir: str = "reports"
        
        # Load configuration if file exists
        if self.config_path.exists():
            self.load_config()
        
        # Override with environment variables
        self.load_environment_variables()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update dataclass instances
        if 'model' in config_data:
            self.model = ModelConfig(**config_data['model'])
        if 'training' in config_data:
            self.training = TrainingConfig(**config_data['training'])
        if 'data' in config_data:
            self.data = DataConfig(**config_data['data'])
        if 'evaluation' in config_data:
            self.evaluation = EvaluationConfig(**config_data['evaluation'])
        
        # Update paths
        for key in ['data_path', 'output_dir', 'checkpoint_dir', 'logs_dir', 'reports_dir']:
            if key in config_data:
                setattr(self, key, config_data[key])
    
    def load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            'FR_DATA_PATH': 'data_path',
            'FR_OUTPUT_DIR': 'output_dir',
            'FR_BATCH_SIZE': ('training', 'batch_size', int),
            'FR_LEARNING_RATE': ('training', 'learning_rate', float),
            'FR_EPOCHS': ('training', 'epochs', int),
            'FR_MODEL_ARCH': ('model', 'architecture', str),
        }
        
        for env_var, mapping in env_mappings.items():
            if env_var in os.environ:
                if isinstance(mapping, str):
                    setattr(self, mapping, os.environ[env_var])
                else:
                    obj, attr, type_func = mapping
                    setattr(getattr(self, obj), attr, type_func(os.environ[env_var]))
    
    def update_from_args(self, args) -> None:
        """Update configuration from command line arguments."""
        if args.data_path:
            self.data_path = args.data_path
        if args.output_dir:
            self.output_dir = args.output_dir
        if args.model:
            self.model.architecture = args.model
        if args.resume:
            self.resume_checkpoint = args.resume
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.data_path:
            raise ValueError("Data path must be specified")
        
        if not Path(self.data_path).exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.training.epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if not 0 < self.data.train_split < 1:
            raise ValueError("Train split must be between 0 and 1")
        
        if self.data.train_split + self.data.test_split != 1.0:
            raise ValueError("Train and test splits must sum to 1.0")
    
    def create_directories(self) -> None:
        """Create necessary output directories."""
        directories = [
            self.output_dir,
            self.checkpoint_dir,
            self.logs_dir,
            self.reports_dir,
            f"{self.reports_dir}/figures"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to YAML file."""
        if output_path is None:
            output_path = f"{self.output_dir}/config_used.yaml"
        
        config_dict = {
            'model': {
                'architecture': self.model.architecture,
                'backbone': self.model.backbone,
                'embedding_size': self.model.embedding_size,
                'pretrained': self.model.pretrained,
                'dropout_rate': self.model.dropout_rate
            },
            'training': {
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'epochs': self.training.epochs,
                'optimizer': self.training.optimizer,
                'scheduler': self.training.scheduler,
                'weight_decay': self.training.weight_decay,
                'gradient_clip': self.training.gradient_clip,
                'freeze_backbone': self.training.freeze_backbone,
                'unfreeze_after_epoch': self.training.unfreeze_after_epoch,
                'use_augmentation': self.training.use_augmentation,
                'augmentation_strength': self.training.augmentation_strength
            },
            'data': {
                'train_split': self.data.train_split,
                'test_split': self.data.test_split,
                'image_size': self.data.image_size,
                'normalize_mean': self.data.normalize_mean,
                'normalize_std': self.data.normalize_std,
                'min_face_size': self.data.min_face_size,
                'blur_threshold': self.data.blur_threshold,
                'illumination_threshold': self.data.illumination_threshold
            },
            'evaluation': {
                'similarity_threshold': self.evaluation.similarity_threshold,
                'distance_metric': self.evaluation.distance_metric,
                'threshold_range': self.evaluation.threshold_range,
                'threshold_steps': self.evaluation.threshold_steps,
                'plot_roc': self.evaluation.plot_roc,
                'plot_precision_recall': self.evaluation.plot_precision_recall,
                'plot_similarity_distribution': self.evaluation.plot_similarity_distribution
            },
            'paths': {
                'data_path': self.data_path,
                'output_dir': self.output_dir,
                'checkpoint_dir': self.checkpoint_dir,
                'logs_dir': self.logs_dir,
                'reports_dir': self.reports_dir
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging."""
        return {
            'model_architecture': self.model.architecture,
            'batch_size': self.training.batch_size,
            'learning_rate': self.training.learning_rate,
            'epochs': self.training.epochs,
            'data_path': self.data_path,
            'output_dir': self.output_dir
        }
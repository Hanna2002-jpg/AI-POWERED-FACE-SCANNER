"""
Model Setup and Architecture Definitions
========================================

Implements various face recognition models including ArcFace, InsightFace, and FaceNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import logging
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

from .utils import get_device


class ArcMarginProduct(nn.Module):
    """ArcFace: Additive Angular Margin Loss for Deep Face Recognition."""
    
    def __init__(self, in_features: int, out_features: int, scale: float = 64.0, 
                 margin: float = 0.5, easy_margin: bool = False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Calculate cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output


class FaceRecognitionModel(nn.Module, ABC):
    """Abstract base class for face recognition models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.embedding_size = config['model']['embedding_size']
        self.num_classes = None  # Will be set during training
        
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings from input."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model."""
        pass


class ArcFaceModel(FaceRecognitionModel):
    """ArcFace model implementation."""
    
    def __init__(self, config: Dict[str, Any], num_classes: Optional[int] = None):
        super().__init__(config)
        self.num_classes = num_classes
        
        # Backbone network
        backbone_name = config['model']['backbone']
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=config['model']['pretrained'])
            backbone_output_size = 2048
        elif backbone_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=config['model']['pretrained'])
            backbone_output_size = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Feature embedding layer
        self.embedding = nn.Sequential(
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(backbone_output_size, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size)
        )
        
        # ArcFace margin layer (only used during training)
        if num_classes is not None:
            self.margin_layer = ArcMarginProduct(
                self.embedding_size, 
                num_classes,
                scale=config['loss']['scale'],
                margin=config['loss']['margin']
            )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalized feature embeddings."""
        # Backbone features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Embedding layer
        embeddings = self.embedding(features)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model."""
        embeddings = self.extract_features(x)
        
        if self.training and labels is not None and hasattr(self, 'margin_layer'):
            # Training mode with ArcFace loss
            output = self.margin_layer(embeddings, labels)
            return output
        else:
            # Inference mode - return embeddings
            return embeddings


class InsightFaceModel(FaceRecognitionModel):
    """InsightFace model implementation (simplified version)."""
    
    def __init__(self, config: Dict[str, Any], num_classes: Optional[int] = None):
        super().__init__(config)
        self.num_classes = num_classes
        
        # Use ResNet backbone
        if config['model']['backbone'] == 'resnet50':
            self.backbone = models.resnet50(pretrained=config['model']['pretrained'])
            backbone_output_size = 2048
        else:
            raise ValueError(f"Unsupported backbone for InsightFace: {config['model']['backbone']}")
        
        # Modify backbone
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Feature layers
        self.features = nn.Sequential(
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(backbone_output_size, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size)
        )
        
        # Classification layer for training
        if num_classes is not None:
            self.classifier = nn.Linear(self.embedding_size, num_classes)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings."""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.features(features)
        return F.normalize(embeddings, p=2, dim=1)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model."""
        embeddings = self.extract_features(x)
        
        if self.training and hasattr(self, 'classifier'):
            # Training mode
            logits = self.classifier(embeddings)
            return logits
        else:
            # Inference mode
            return embeddings


class FaceNetModel(FaceRecognitionModel):
    """FaceNet model implementation using triplet loss."""
    
    def __init__(self, config: Dict[str, Any], num_classes: Optional[int] = None):
        super().__init__(config)
        
        # Use pre-trained FaceNet if available, otherwise use ResNet
        try:
            from facenet_pytorch import InceptionResnetV1
            self.backbone = InceptionResnetV1(pretrained='vggface2')
            backbone_output_size = 512
        except ImportError:
            # Fallback to ResNet
            self.backbone = models.resnet50(pretrained=config['model']['pretrained'])
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            backbone_output_size = 2048
        
        # Embedding layer
        if backbone_output_size != self.embedding_size:
            self.embedding = nn.Sequential(
                nn.Dropout(config['model']['dropout_rate']),
                nn.Linear(backbone_output_size, self.embedding_size),
                nn.BatchNorm1d(self.embedding_size)
            )
        else:
            self.embedding = nn.Identity()
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings."""
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, p=2, dim=1)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model."""
        return self.extract_features(x)


class ModelFactory:
    """Factory class for creating face recognition models."""
    
    @staticmethod
    def create_model(config: Dict[str, Any], num_classes: Optional[int] = None) -> FaceRecognitionModel:
        """Create a face recognition model based on configuration."""
        architecture = config['model']['architecture'].lower()
        
        if architecture == 'arcface':
            return ArcFaceModel(config, num_classes)
        elif architecture == 'insightface':
            return InsightFaceModel(config, num_classes)
        elif architecture == 'facenet':
            return FaceNetModel(config, num_classes)
        else:
            raise ValueError(f"Unsupported model architecture: {architecture}")


class TripletLoss(nn.Module):
    """Triplet loss for FaceNet training."""
    
    def __init__(self, margin: float = 0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss."""
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class CenterLoss(nn.Module):
    """Center loss for face recognition."""
    
    def __init__(self, num_classes: int, feat_dim: int, device: torch.device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute center loss."""
        batch_size = features.size(0)
        
        # Compute distances to centers
        centers_batch = self.centers.index_select(0, labels.long())
        criterion = nn.MSELoss()
        center_loss = criterion(features, centers_batch)
        
        return center_loss


def load_pretrained_model(model_path: str, config: Dict[str, Any], 
                         num_classes: Optional[int] = None) -> FaceRecognitionModel:
    """Load a pre-trained model from checkpoint."""
    device = get_device(config)
    
    # Create model
    model = ModelFactory.create_model(config, num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def save_model_checkpoint(model: FaceRecognitionModel, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, accuracy: float, 
                         checkpoint_path: str) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    torch.save(checkpoint, checkpoint_path)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: FaceRecognitionModel) -> None:
    """Freeze backbone parameters for transfer learning."""
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: FaceRecognitionModel) -> None:
    """Unfreeze backbone parameters."""
    for param in model.backbone.parameters():
        param.requires_grad = True


def get_model_summary(model: FaceRecognitionModel) -> Dict[str, Any]:
    """Get model summary information."""
    total_params = count_parameters(model)
    
    summary = {
        'architecture': model.config['model']['architecture'],
        'backbone': model.config['model']['backbone'],
        'embedding_size': model.embedding_size,
        'total_parameters': total_params,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    return summary
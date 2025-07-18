"""
Baseline Evaluation Module
==========================

Evaluates pre-trained face recognition models before fine-tuning to establish baseline performance.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm
import json
from pathlib import Path

from .model_setup import ModelFactory, load_pretrained_model
from .utils import (setup_logging, get_device, save_metrics, 
                   load_metrics, plot_training_curves)


class FaceDataset(Dataset):
    """Dataset class for face recognition evaluation."""
    
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        self.labels = []
        self.label_to_idx = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset and create label mappings."""
        idx = 0
        for identity_dir in os.listdir(self.data_path):
            identity_path = os.path.join(self.data_path, identity_dir)
            if not os.path.isdir(identity_path):
                continue
            
            self.label_to_idx[identity_dir] = idx
            
            for image_file in os.listdir(identity_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(identity_path, image_file)
                    self.samples.append(image_path)
                    self.labels.append(idx)
            
            idx += 1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label, image_path


class BaselineEvaluator:
    """Baseline evaluation for pre-trained models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging()
        self.device = get_device(config)
        
        # Initialize models to evaluate
        self.models_to_evaluate = self._get_models_to_evaluate()
    
    def _get_models_to_evaluate(self) -> List[str]:
        """Get list of model architectures to evaluate."""
        # For baseline, we'll evaluate the configured model
        return [self.config['model']['architecture']]
    
    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all baseline models."""
        self.logger.info("Starting baseline evaluation...")
        
        results = {}
        
        for model_name in self.models_to_evaluate:
            self.logger.info(f"Evaluating {model_name} model...")
            
            # Update config for current model
            model_config = self.config.copy()
            model_config['model']['architecture'] = model_name
            
            try:
                model_results = self._evaluate_single_model(model_config)
                results[model_name] = model_results
                
                self.logger.info(f"{model_name} evaluation completed")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Save baseline results
        save_metrics(results, 'baseline_results.json', self.config)
        
        # Generate comparison plots
        self._plot_model_comparison(results)
        
        self.logger.info("Baseline evaluation completed")
        return results
    
    def _evaluate_single_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single model architecture."""
        # Create model (without num_classes for inference)
        model = ModelFactory.create_model(model_config, num_classes=None)
        model.to(self.device)
        model.eval()
        
        # Load evaluation dataset
        eval_dataset = FaceDataset(
            self.config['dataset']['eval_path'],
            transform=None  # We'll handle transforms manually
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        # Extract embeddings
        embeddings, labels, image_paths = self._extract_embeddings(model, eval_loader)
        
        # Compute similarity metrics
        similarity_results = self._compute_similarity_metrics(embeddings, labels)
        
        # Generate evaluation plots
        plot_results = self._generate_evaluation_plots(
            embeddings, labels, model_config['model']['architecture']
        )
        
        # Combine results
        results = {
            'model_info': {
                'architecture': model_config['model']['architecture'],
                'backbone': model_config['model']['backbone'],
                'embedding_size': model_config['model']['embedding_size']
            },
            'dataset_info': {
                'num_samples': len(embeddings),
                'num_identities': len(np.unique(labels))
            },
            'metrics': similarity_results,
            'plots': plot_results
        }
        
        return results
    
    def _extract_embeddings(self, model: torch.nn.Module, 
                           dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract embeddings from the model."""
        embeddings = []
        labels = []
        image_paths = []
        
        with torch.no_grad():
            for batch_images, batch_labels, batch_paths in tqdm(dataloader, desc="Extracting embeddings"):
                # Preprocess images
                batch_images = self._preprocess_batch(batch_images)
                batch_images = batch_images.to(self.device)
                
                # Extract features
                batch_embeddings = model.extract_features(batch_images)
                
                embeddings.append(batch_embeddings.cpu().numpy())
                labels.append(batch_labels.numpy())
                image_paths.extend(batch_paths)
        
        embeddings = np.vstack(embeddings)
        labels = np.hstack(labels)
        
        return embeddings, labels, image_paths
    
    def _preprocess_batch(self, batch_images: torch.Tensor) -> torch.Tensor:
        """Preprocess batch of images."""
        # Convert to float and normalize
        batch_images = batch_images.float() / 255.0
        
        # Apply normalization
        mean = torch.tensor(self.config['preprocessing']['normalization']['mean'])
        std = torch.tensor(self.config['preprocessing']['normalization']['std'])
        
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        
        batch_images = (batch_images - mean) / std
        
        return batch_images
    
    def _compute_similarity_metrics(self, embeddings: np.ndarray, 
                                   labels: np.ndarray) -> Dict[str, Any]:
        """Compute similarity-based evaluation metrics."""
        self.logger.info("Computing similarity metrics...")
        
        # Generate pairs for evaluation
        same_pairs, diff_pairs = self._generate_evaluation_pairs(embeddings, labels)
        
        results = {}
        
        # Evaluate different similarity metrics
        for metric_name in self.config['evaluation']['similarity_metrics']:
            metric_results = self._evaluate_similarity_metric(
                same_pairs, diff_pairs, metric_name
            )
            results[metric_name] = metric_results
        
        return results
    
    def _generate_evaluation_pairs(self, embeddings: np.ndarray, 
                                  labels: np.ndarray) -> Tuple[List[Tuple], List[Tuple]]:
        """Generate same-identity and different-identity pairs."""
        same_pairs = []
        diff_pairs = []
        
        n_samples = len(embeddings)
        
        # Generate pairs
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if labels[i] == labels[j]:
                    same_pairs.append((embeddings[i], embeddings[j]))
                else:
                    diff_pairs.append((embeddings[i], embeddings[j]))
        
        # Limit number of pairs for efficiency
        max_pairs = 10000
        if len(same_pairs) > max_pairs:
            indices = np.random.choice(len(same_pairs), max_pairs, replace=False)
            same_pairs = [same_pairs[i] for i in indices]
        
        if len(diff_pairs) > max_pairs:
            indices = np.random.choice(len(diff_pairs), max_pairs, replace=False)
            diff_pairs = [diff_pairs[i] for i in indices]
        
        return same_pairs, diff_pairs
    
    def _evaluate_similarity_metric(self, same_pairs: List[Tuple], diff_pairs: List[Tuple],
                                   metric_name: str) -> Dict[str, Any]:
        """Evaluate a specific similarity metric."""
        # Compute similarities
        same_similarities = []
        diff_similarities = []
        
        for emb1, emb2 in same_pairs:
            sim = self._compute_similarity(emb1, emb2, metric_name)
            same_similarities.append(sim)
        
        for emb1, emb2 in diff_pairs:
            sim = self._compute_similarity(emb1, emb2, metric_name)
            diff_similarities.append(sim)
        
        same_similarities = np.array(same_similarities)
        diff_similarities = np.array(diff_similarities)
        
        # Create labels for ROC computation
        y_true = np.concatenate([np.ones(len(same_similarities)), np.zeros(len(diff_similarities))])
        y_scores = np.concatenate([same_similarities, diff_similarities])
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Compute precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # Find optimal threshold (Youden's index)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Compute accuracy at optimal threshold
        predictions = (y_scores >= optimal_threshold).astype(int)
        accuracy = np.mean(predictions == y_true)
        
        # Equal Error Rate
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = fpr[eer_idx]
        
        results = {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'accuracy': float(accuracy),
            'eer': float(eer),
            'optimal_threshold': float(optimal_threshold),
            'same_similarities_stats': {
                'mean': float(np.mean(same_similarities)),
                'std': float(np.std(same_similarities)),
                'min': float(np.min(same_similarities)),
                'max': float(np.max(same_similarities))
            },
            'diff_similarities_stats': {
                'mean': float(np.mean(diff_similarities)),
                'std': float(np.std(diff_similarities)),
                'min': float(np.min(diff_similarities)),
                'max': float(np.max(diff_similarities))
            }
        }
        
        return results
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray, metric: str) -> float:
        """Compute similarity between two embeddings."""
        if metric == 'cosine':
            return cosine_similarity([emb1], [emb2])[0, 0]
        elif metric == 'euclidean':
            return -euclidean_distances([emb1], [emb2])[0, 0]  # Negative for similarity
        elif metric == 'manhattan':
            return -np.sum(np.abs(emb1 - emb2))  # Negative for similarity
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def _generate_evaluation_plots(self, embeddings: np.ndarray, labels: np.ndarray,
                                  model_name: str) -> Dict[str, str]:
        """Generate evaluation plots."""
        plot_files = {}
        
        # ROC curves for different metrics
        self._plot_roc_curves(embeddings, labels, model_name)
        plot_files['roc_curves'] = f"{self.config['paths']['plots']}/baseline_roc_{model_name}.png"
        
        # Similarity distributions
        self._plot_similarity_distributions(embeddings, labels, model_name)
        plot_files['similarity_dist'] = f"{self.config['paths']['plots']}/baseline_similarity_{model_name}.png"
        
        # Embedding visualization (t-SNE)
        self._plot_embedding_visualization(embeddings, labels, model_name)
        plot_files['embedding_viz'] = f"{self.config['paths']['plots']}/baseline_tsne_{model_name}.png"
        
        return plot_files
    
    def _plot_roc_curves(self, embeddings: np.ndarray, labels: np.ndarray, model_name: str):
        """Plot ROC curves for different similarity metrics."""
        plt.figure(figsize=(10, 8))
        
        same_pairs, diff_pairs = self._generate_evaluation_pairs(embeddings, labels)
        
        for metric_name in self.config['evaluation']['similarity_metrics']:
            # Compute similarities
            same_sims = [self._compute_similarity(p[0], p[1], metric_name) for p in same_pairs]
            diff_sims = [self._compute_similarity(p[0], p[1], metric_name) for p in diff_pairs]
            
            # Create labels and scores
            y_true = np.concatenate([np.ones(len(same_sims)), np.zeros(len(diff_sims))])
            y_scores = np.concatenate([same_sims, diff_sims])
            
            # Compute ROC
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{metric_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name} (Baseline)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f"{self.config['paths']['plots']}/baseline_roc_{model_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_similarity_distributions(self, embeddings: np.ndarray, labels: np.ndarray, 
                                     model_name: str):
        """Plot similarity distributions for same vs different identities."""
        same_pairs, diff_pairs = self._generate_evaluation_pairs(embeddings, labels)
        
        fig, axes = plt.subplots(1, len(self.config['evaluation']['similarity_metrics']), 
                                figsize=(15, 5))
        
        if len(self.config['evaluation']['similarity_metrics']) == 1:
            axes = [axes]
        
        for idx, metric_name in enumerate(self.config['evaluation']['similarity_metrics']):
            same_sims = [self._compute_similarity(p[0], p[1], metric_name) for p in same_pairs]
            diff_sims = [self._compute_similarity(p[0], p[1], metric_name) for p in diff_pairs]
            
            axes[idx].hist(same_sims, bins=50, alpha=0.7, label='Same Identity', density=True)
            axes[idx].hist(diff_sims, bins=50, alpha=0.7, label='Different Identity', density=True)
            axes[idx].set_xlabel(f'{metric_name.capitalize()} Similarity')
            axes[idx].set_ylabel('Density')
            axes[idx].set_title(f'{metric_name.capitalize()} Distribution')
            axes[idx].legend()
            axes[idx].grid(True)
        
        plt.suptitle(f'Similarity Distributions - {model_name} (Baseline)')
        plt.tight_layout()
        
        plt.savefig(f"{self.config['paths']['plots']}/baseline_similarity_{model_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_embedding_visualization(self, embeddings: np.ndarray, labels: np.ndarray, 
                                    model_name: str):
        """Plot t-SNE visualization of embeddings."""
        try:
            from sklearn.manifold import TSNE
            
            # Sample embeddings for visualization (t-SNE is slow)
            max_samples = 1000
            if len(embeddings) > max_samples:
                indices = np.random.choice(len(embeddings), max_samples, replace=False)
                embeddings_sample = embeddings[indices]
                labels_sample = labels[indices]
            else:
                embeddings_sample = embeddings
                labels_sample = labels
            
            # Compute t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = tsne.fit_transform(embeddings_sample)
            
            # Plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=labels_sample, cmap='tab10', alpha=0.7)
            plt.colorbar(scatter)
            plt.title(f't-SNE Visualization - {model_name} (Baseline)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
            plt.savefig(f"{self.config['paths']['plots']}/baseline_tsne_{model_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            self.logger.warning("scikit-learn not available for t-SNE visualization")
    
    def _plot_model_comparison(self, results: Dict[str, Any]):
        """Plot comparison between different models."""
        if len(results) < 2:
            return
        
        metrics_to_compare = ['roc_auc', 'accuracy', 'eer']
        similarity_metrics = self.config['evaluation']['similarity_metrics']
        
        fig, axes = plt.subplots(1, len(metrics_to_compare), figsize=(15, 5))
        
        for idx, metric in enumerate(metrics_to_compare):
            model_names = []
            metric_values = []
            
            for model_name, model_results in results.items():
                if 'error' in model_results:
                    continue
                
                model_names.append(model_name)
                # Use cosine similarity results as default
                if 'cosine' in model_results['metrics']:
                    metric_values.append(model_results['metrics']['cosine'][metric])
                else:
                    # Use first available metric
                    first_metric = list(model_results['metrics'].keys())[0]
                    metric_values.append(model_results['metrics'][first_metric][metric])
            
            axes[idx].bar(model_names, metric_values)
            axes[idx].set_title(f'{metric.upper()} Comparison')
            axes[idx].set_ylabel(metric.upper())
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Baseline Model Comparison')
        plt.tight_layout()
        
        plt.savefig(f"{self.config['paths']['plots']}/baseline_model_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function for standalone baseline evaluation."""
    import argparse
    from .utils import load_config, create_directories, set_random_seeds
    
    parser = argparse.ArgumentParser(description="Face Recognition Baseline Evaluation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup
    set_random_seeds(42)
    create_directories(config)
    
    # Run baseline evaluation
    evaluator = BaselineEvaluator(config)
    results = evaluator.evaluate_all_models()
    
    # Print summary
    print("\nBaseline Evaluation Results:")
    print("=" * 50)
    
    for model_name, model_results in results.items():
        if 'error' in model_results:
            print(f"{model_name}: Error - {model_results['error']}")
            continue
        
        print(f"\n{model_name}:")
        for metric_name, metric_results in model_results['metrics'].items():
            print(f"  {metric_name}:")
            print(f"    ROC AUC: {metric_results['roc_auc']:.4f}")
            print(f"    Accuracy: {metric_results['accuracy']:.4f}")
            print(f"    EER: {metric_results['eer']:.4f}")


if __name__ == "__main__":
    main()
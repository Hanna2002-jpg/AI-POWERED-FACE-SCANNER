"""
Data Preprocessing Module
=========================

Handles dataset preparation, face detection, alignment, and train/test splitting
with strict integrity verification.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision import transforms
from PIL import Image
import logging
from tqdm import tqdm
import shutil

from .utils import (setup_logging, save_split_metadata, compute_dataset_hash,
                   verify_data_integrity, save_metrics)


class FaceDetector:
    """Face detection and alignment using MTCNN or OpenCV."""
    
    def __init__(self, method: str = "opencv", confidence_threshold: float = 0.9):
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        if method == "opencv":
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        elif method == "mtcnn":
            try:
                from facenet_pytorch import MTCNN
                self.mtcnn = MTCNN(keep_all=False, post_process=False)
            except ImportError:
                self.logger.warning("MTCNN not available, falling back to OpenCV")
                self.method = "opencv"
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
    
    def detect_and_align(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Detect and align face in image."""
        if self.method == "mtcnn":
            return self._detect_mtcnn(image)
        else:
            return self._detect_opencv(image)
    
    def _detect_opencv(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Face detection using OpenCV Haar cascades."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return image, False
        
        # Use the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Add padding
        padding = int(0.2 * min(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        face_crop = image[y:y+h, x:x+w]
        return face_crop, True
    
    def _detect_mtcnn(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Face detection using MTCNN."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Detect face
            face_tensor = self.mtcnn(pil_image)
            
            if face_tensor is None:
                return image, False
            
            # Convert back to numpy array
            face_array = face_tensor.permute(1, 2, 0).numpy()
            face_array = (face_array * 255).astype(np.uint8)
            
            return face_array, True
        except Exception as e:
            self.logger.warning(f"MTCNN detection failed: {e}")
            return self._detect_opencv(image)


class DataPreprocessor:
    """Main data preprocessing class."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging()
        self.face_detector = FaceDetector(
            method=config['preprocessing']['alignment_method'],
            confidence_threshold=config['preprocessing']['face_detection_confidence']
        )
        
        # Setup transforms
        self.train_transform = self._get_train_transforms()
        self.eval_transform = self._get_eval_transforms()
    
    def _get_train_transforms(self) -> A.Compose:
        """Get training data augmentation transforms."""
        aug_config = self.config['preprocessing']['augmentation']
        
        transforms_list = [
            A.Resize(*self.config['dataset']['image_size']),
            A.HorizontalFlip(p=aug_config['horizontal_flip']),
            A.Rotate(limit=aug_config['rotation_range'], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config['brightness_range'],
                contrast_limit=aug_config['contrast_range'],
                p=0.5
            ),
            A.Normalize(
                mean=self.config['preprocessing']['normalization']['mean'],
                std=self.config['preprocessing']['normalization']['std']
            ),
            ToTensorV2()
        ]
        
        return A.Compose(transforms_list)
    
    def _get_eval_transforms(self) -> A.Compose:
        """Get evaluation transforms (no augmentation)."""
        transforms_list = [
            A.Resize(*self.config['dataset']['image_size']),
            A.Normalize(
                mean=self.config['preprocessing']['normalization']['mean'],
                std=self.config['preprocessing']['normalization']['std']
            ),
            ToTensorV2()
        ]
        
        return A.Compose(transforms_list)
    
    def process_dataset(self, raw_data_path: str) -> None:
        """Process raw dataset and create train/eval splits."""
        self.logger.info("Starting dataset processing...")
        
        # Discover and validate dataset
        image_files, identity_mapping = self._discover_dataset(raw_data_path)
        
        # Quality assessment
        quality_stats = self._assess_image_quality(image_files)
        
        # Create train/eval split
        train_files, eval_files = self._create_split(image_files, identity_mapping)
        
        # Process and save images
        self._process_and_save_images(train_files, eval_files)
        
        # Save metadata and integrity information
        self._save_processing_metadata(train_files, eval_files, quality_stats)
        
        self.logger.info("Dataset processing completed successfully!")
    
    def _discover_dataset(self, raw_data_path: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Discover all images and create identity mapping."""
        self.logger.info("Discovering dataset structure...")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        identity_mapping = {}
        
        for root, dirs, files in os.walk(raw_data_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    file_path = os.path.join(root, file)
                    image_files.append(file_path)
                    
                    # Extract identity from directory structure
                    identity = os.path.basename(root)
                    if identity not in identity_mapping:
                        identity_mapping[identity] = []
                    identity_mapping[identity].append(file_path)
        
        # Filter identities with minimum images
        min_images = self.config['dataset']['min_images_per_identity']
        filtered_mapping = {
            identity: files for identity, files in identity_mapping.items()
            if len(files) >= min_images
        }
        
        # Update image files list
        filtered_files = []
        for files in filtered_mapping.values():
            filtered_files.extend(files)
        
        self.logger.info(f"Found {len(filtered_files)} images from {len(filtered_mapping)} identities")
        
        return filtered_files, filtered_mapping
    
    def _assess_image_quality(self, image_files: List[str]) -> Dict[str, Any]:
        """Assess image quality metrics."""
        self.logger.info("Assessing image quality...")
        
        quality_stats = {
            'total_images': len(image_files),
            'valid_faces': 0,
            'invalid_faces': 0,
            'blur_scores': [],
            'brightness_scores': [],
            'resolution_stats': []
        }
        
        for image_path in tqdm(image_files, desc="Quality assessment"):
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Face detection
                _, face_detected = self.face_detector.detect_and_align(image)
                if face_detected:
                    quality_stats['valid_faces'] += 1
                else:
                    quality_stats['invalid_faces'] += 1
                
                # Blur detection (Laplacian variance)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                quality_stats['blur_scores'].append(blur_score)
                
                # Brightness assessment
                brightness = np.mean(gray)
                quality_stats['brightness_scores'].append(brightness)
                
                # Resolution
                h, w = image.shape[:2]
                quality_stats['resolution_stats'].append((w, h))
                
            except Exception as e:
                self.logger.warning(f"Error processing {image_path}: {e}")
        
        # Compute statistics
        quality_stats['avg_blur'] = np.mean(quality_stats['blur_scores'])
        quality_stats['avg_brightness'] = np.mean(quality_stats['brightness_scores'])
        quality_stats['face_detection_rate'] = quality_stats['valid_faces'] / quality_stats['total_images']
        
        self.logger.info(f"Quality assessment: {quality_stats['face_detection_rate']:.2%} face detection rate")
        
        return quality_stats
    
    def _create_split(self, image_files: List[str], 
                     identity_mapping: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
        """Create stratified train/eval split ensuring no identity overlap."""
        self.logger.info("Creating train/eval split...")
        
        identities = list(identity_mapping.keys())
        train_split = self.config['dataset']['train_split']
        
        # Split identities (not images) to ensure no overlap
        train_identities, eval_identities = train_test_split(
            identities, 
            train_size=train_split,
            random_state=42,
            shuffle=True
        )
        
        # Collect files for each split
        train_files = []
        eval_files = []
        
        for identity in train_identities:
            train_files.extend(identity_mapping[identity])
        
        for identity in eval_identities:
            eval_files.extend(identity_mapping[identity])
        
        self.logger.info(f"Split created: {len(train_files)} train, {len(eval_files)} eval images")
        self.logger.info(f"Identity split: {len(train_identities)} train, {len(eval_identities)} eval identities")
        
        # Verify no overlap
        train_set = set(train_files)
        eval_set = set(eval_files)
        assert len(train_set.intersection(eval_set)) == 0, "Data leakage detected!"
        
        return train_files, eval_files
    
    def _process_and_save_images(self, train_files: List[str], eval_files: List[str]) -> None:
        """Process and save images to train/eval directories."""
        self.logger.info("Processing and saving images...")
        
        # Process training images
        self._process_image_set(train_files, self.config['dataset']['train_path'], "train")
        
        # Process evaluation images
        self._process_image_set(eval_files, self.config['dataset']['eval_path'], "eval")
    
    def _process_image_set(self, image_files: List[str], output_path: str, split_name: str) -> None:
        """Process a set of images and save to output directory."""
        processed_count = 0
        
        for image_path in tqdm(image_files, desc=f"Processing {split_name} images"):
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Detect and align face
                face_image, face_detected = self.face_detector.detect_and_align(image)
                
                if not face_detected:
                    # Use original image if face detection fails
                    face_image = image
                
                # Resize to target size
                target_size = tuple(self.config['dataset']['image_size'])
                face_image = cv2.resize(face_image, target_size)
                
                # Create output directory structure
                relative_path = os.path.relpath(image_path, self.config['dataset']['raw_path'])
                output_file_path = os.path.join(output_path, relative_path)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                
                # Save processed image
                cv2.imwrite(output_file_path, face_image)
                processed_count += 1
                
            except Exception as e:
                self.logger.warning(f"Error processing {image_path}: {e}")
        
        self.logger.info(f"Processed {processed_count} {split_name} images")
    
    def _save_processing_metadata(self, train_files: List[str], eval_files: List[str],
                                 quality_stats: Dict[str, Any]) -> None:
        """Save processing metadata and integrity information."""
        self.logger.info("Saving processing metadata...")
        
        # Save split metadata
        metadata_path = os.path.join(self.config['paths']['results'], 'split_metadata.json')
        save_split_metadata(train_files, eval_files, metadata_path)
        
        # Save quality assessment
        quality_path = os.path.join(self.config['paths']['metrics'], 'quality_assessment.json')
        save_metrics(quality_stats, 'quality_assessment.json', self.config)
        
        # Compute and save dataset hashes
        dataset_hashes = {
            'raw_dataset_hash': compute_dataset_hash(self.config['dataset']['raw_path']),
            'train_dataset_hash': compute_dataset_hash(self.config['dataset']['train_path']),
            'eval_dataset_hash': compute_dataset_hash(self.config['dataset']['eval_path'])
        }
        
        save_metrics(dataset_hashes, 'dataset_hashes.json', self.config)
        
        self.logger.info("Metadata saved successfully")
    
    def verify_preprocessing_integrity(self) -> bool:
        """Verify preprocessing integrity."""
        self.logger.info("Verifying preprocessing integrity...")
        
        metadata_path = os.path.join(self.config['paths']['results'], 'split_metadata.json')
        
        if not os.path.exists(metadata_path):
            self.logger.error("Split metadata not found")
            return False
        
        integrity_check = verify_data_integrity(metadata_path)
        
        if integrity_check:
            self.logger.info("Data integrity verification passed")
        else:
            self.logger.error("Data integrity verification failed")
        
        return integrity_check


def main():
    """Main function for standalone preprocessing."""
    import argparse
    from .utils import load_config, create_directories, set_random_seeds
    
    parser = argparse.ArgumentParser(description="Face Recognition Data Preprocessing")
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    parser.add_argument('--raw-data', type=str, required=True, help='Raw dataset path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['dataset']['raw_path'] = args.raw_data
    
    # Setup
    set_random_seeds(42)
    create_directories(config)
    
    # Process dataset
    preprocessor = DataPreprocessor(config)
    preprocessor.process_dataset(args.raw_data)
    
    # Verify integrity
    preprocessor.verify_preprocessing_integrity()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Dataset Setup Script
===================

Helper script to organize and validate dataset structure before preprocessing.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List
import logging

def setup_logging():
    """Setup logging for dataset operations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def validate_dataset_structure(raw_data_path: str) -> Dict[str, int]:
    """Validate and analyze dataset structure."""
    logger = setup_logging()
    
    if not os.path.exists(raw_data_path):
        logger.error(f"Raw data path does not exist: {raw_data_path}")
        return {}
    
    stats = {
        'total_identities': 0,
        'total_images': 0,
        'min_images_per_identity': float('inf'),
        'max_images_per_identity': 0,
        'identities_with_insufficient_images': 0
    }
    
    identity_counts = {}
    
    for identity_dir in os.listdir(raw_data_path):
        identity_path = os.path.join(raw_data_path, identity_dir)
        
        if not os.path.isdir(identity_path):
            continue
        
        # Count images in this identity folder
        image_count = 0
        for file in os.listdir(identity_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_count += 1
        
        if image_count > 0:
            stats['total_identities'] += 1
            stats['total_images'] += image_count
            identity_counts[identity_dir] = image_count
            
            stats['min_images_per_identity'] = min(stats['min_images_per_identity'], image_count)
            stats['max_images_per_identity'] = max(stats['max_images_per_identity'], image_count)
            
            if image_count < 2:  # Minimum required for train/test split
                stats['identities_with_insufficient_images'] += 1
    
    # Log statistics
    logger.info("Dataset Analysis:")
    logger.info(f"  Total identities: {stats['total_identities']}")
    logger.info(f"  Total images: {stats['total_images']}")
    logger.info(f"  Images per identity - Min: {stats['min_images_per_identity']}, Max: {stats['max_images_per_identity']}")
    logger.info(f"  Average images per identity: {stats['total_images'] / max(stats['total_identities'], 1):.1f}")
    
    if stats['identities_with_insufficient_images'] > 0:
        logger.warning(f"  {stats['identities_with_insufficient_images']} identities have insufficient images (< 2)")
    
    return stats

def create_sample_dataset(raw_data_path: str):
    """Create a sample dataset structure for demonstration."""
    logger = setup_logging()
    
    logger.info("Creating sample dataset structure...")
    
    # Create sample identities
    sample_identities = ['person_001', 'person_002', 'person_003', 'person_004', 'person_005']
    
    for identity in sample_identities:
        identity_path = os.path.join(raw_data_path, identity)
        os.makedirs(identity_path, exist_ok=True)
        
        # Create placeholder files to show structure
        readme_content = f"""# {identity} Image Directory

Place images of {identity} in this directory.

Supported formats:
- .jpg, .jpeg
- .png
- .bmp

Minimum 2 images required per identity for train/test split.
Recommended: 5+ images per identity for better training.

Example files:
- {identity}_001.jpg
- {identity}_002.jpg
- {identity}_003.png
"""
        
        with open(os.path.join(identity_path, 'README.md'), 'w') as f:
            f.write(readme_content)
    
    logger.info(f"Sample dataset structure created in {raw_data_path}")
    logger.info("Please replace the README.md files with actual images of each person.")

def main():
    """Main function to setup dataset."""
    raw_data_path = "face_recognition_project/data/raw"
    
    logger = setup_logging()
    logger.info("Starting dataset setup...")
    
    # Create directories
    os.makedirs(raw_data_path, exist_ok=True)
    
    # Check if dataset already exists
    if os.listdir(raw_data_path):
        logger.info("Existing data found in raw directory. Validating structure...")
        stats = validate_dataset_structure(raw_data_path)
        
        if stats['total_identities'] == 0:
            logger.warning("No valid identities found. Creating sample structure...")
            create_sample_dataset(raw_data_path)
    else:
        logger.info("No data found. Creating sample dataset structure...")
        create_sample_dataset(raw_data_path)
    
    logger.info("Dataset setup complete!")
    logger.info(f"Next steps:")
    logger.info(f"1. Add your images to {raw_data_path}/[identity_name]/")
    logger.info(f"2. Run preprocessing: python -m face_recognition_project.src.main --raw-data {raw_data_path} --stage preprocess")

if __name__ == "__main__":
    main()
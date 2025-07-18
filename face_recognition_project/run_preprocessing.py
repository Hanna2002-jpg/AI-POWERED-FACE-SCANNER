#!/usr/bin/env python3
"""
Preprocessing Runner Script
==========================

Simplified script to run the data preprocessing pipeline.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_preprocessing():
    """Run the data preprocessing pipeline."""
    
    # Set up paths
    project_root = Path("face_recognition_project")
    raw_data_path = project_root / "data" / "raw"
    
    print("Face Recognition Dataset Preprocessing")
    print("=" * 50)
    
    # Check if raw data directory exists and has content
    if not raw_data_path.exists():
        print(f"❌ Raw data directory not found: {raw_data_path}")
        print("Please run setup_dataset.py first to create the directory structure.")
        return False
    
    # Count identities and images
    identity_count = 0
    total_images = 0
    
    for identity_dir in raw_data_path.iterdir():
        if identity_dir.is_dir() and not identity_dir.name.startswith('.'):
            image_count = len([f for f in identity_dir.iterdir() 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
            if image_count > 0:
                identity_count += 1
                total_images += image_count
                print(f"  📁 {identity_dir.name}: {image_count} images")
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Total identities: {identity_count}")
    print(f"  Total images: {total_images}")
    
    if identity_count == 0:
        print("❌ No valid identities found with images.")
        print("Please add images to the identity folders in data/raw/")
        return False
    
    if total_images < 10:
        print("⚠️  Warning: Very few images found. Consider adding more for better results.")
    
    # Run preprocessing
    print(f"\n🚀 Starting preprocessing pipeline...")
    
    try:
        # Change to project directory
        os.chdir("face_recognition_project")
        
        # Run the preprocessing command
        cmd = [
            sys.executable, "-m", "src.data_preprocessing",
            "--config", "config.yaml",
            "--raw-data", "data/raw"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Preprocessing completed successfully!")
            print("\nOutput:")
            print(result.stdout)
            
            # Check if processed data was created
            if Path("data/train").exists() and Path("data/evaluation").exists():
                train_count = len(list(Path("data/train").rglob("*.jpg"))) + len(list(Path("data/train").rglob("*.png")))
                eval_count = len(list(Path("data/evaluation").rglob("*.jpg"))) + len(list(Path("data/evaluation").rglob("*.png")))
                
                print(f"\n📈 Processing Results:")
                print(f"  Training images: {train_count}")
                print(f"  Evaluation images: {eval_count}")
                print(f"  Train/Eval ratio: {train_count/(train_count+eval_count)*100:.1f}%/{eval_count/(train_count+eval_count)*100:.1f}%")
            
            return True
        else:
            print("❌ Preprocessing failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running preprocessing: {e}")
        return False
    finally:
        # Change back to original directory
        os.chdir("..")

def main():
    """Main function."""
    print("Face Recognition Project - Dataset Preprocessing")
    print("=" * 60)
    
    success = run_preprocessing()
    
    if success:
        print("\n🎉 Preprocessing completed successfully!")
        print("\nNext steps:")
        print("1. Run baseline evaluation: python -m face_recognition_project.src.baseline_evaluation")
        print("2. Start training: python -m face_recognition_project.src.fine_tuning")
        print("3. Run full pipeline: python -m face_recognition_project.src.main --raw-data face_recognition_project/data/raw")
    else:
        print("\n❌ Preprocessing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
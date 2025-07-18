#!/usr/bin/env python3
"""
Face Recognition Project - Main Execution Script
================================================

Complete face recognition system implementation with fine-tuning and evaluation.
Designed for AI Developer Intern assessment with production-ready code quality.

Author: AI Developer Intern Candidate
Date: January 2024
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from src.config import Config
from src.train import TrainingPipeline
from data.dataset_handler import DatasetHandler
from models.model_utils import ModelFactory
from evaluation.baseline_evaluation import BaselineEvaluator
from evaluation.post_training_eval import PostTrainingEvaluator
from reports.report_generator import ReportGenerator


def setup_logging(log_level: str = "INFO") -> None:
    """Setup comprehensive logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/logs/face_recognition.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main execution function with comprehensive CLI interface."""
    parser = argparse.ArgumentParser(
        description="Face Recognition System - Complete Implementation & Evaluation"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--stage', 
        type=str, 
        choices=['preprocess', 'baseline', 'train', 'evaluate', 'report', 'all'],
        default='all',
        help='Pipeline stage to execute'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['arcface', 'insightface', 'facenet'],
        default='arcface',
        help='Model architecture to use'
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        required=True,
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Resume training from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config(args.config)
        config.update_from_args(args)
        
        logger.info("Starting Face Recognition System Pipeline")
        logger.info(f"Configuration: {config.get_summary()}")
        
        # Initialize components
        dataset_handler = DatasetHandler(config)
        model_factory = ModelFactory(config)
        
        # Execute pipeline stages
        if args.stage in ['preprocess', 'all']:
            logger.info("Stage 1: Data Preprocessing")
            dataset_handler.load_and_preprocess()
            dataset_handler.create_train_test_split()
            dataset_handler.verify_data_integrity()
            logger.info("Data preprocessing completed successfully")
        
        if args.stage in ['baseline', 'all']:
            logger.info("Stage 2: Baseline Evaluation")
            baseline_evaluator = BaselineEvaluator(config)
            baseline_results = baseline_evaluator.evaluate_pretrained_models()
            logger.info(f"Baseline evaluation completed: {baseline_results}")
        
        if args.stage in ['train', 'all']:
            logger.info("Stage 3: Model Training")
            training_pipeline = TrainingPipeline(config)
            training_results = training_pipeline.train_model()
            logger.info(f"Training completed: {training_results}")
        
        if args.stage in ['evaluate', 'all']:
            logger.info("Stage 4: Post-Training Evaluation")
            post_evaluator = PostTrainingEvaluator(config)
            evaluation_results = post_evaluator.comprehensive_evaluation()
            logger.info(f"Evaluation completed: {evaluation_results}")
        
        if args.stage in ['report', 'all']:
            logger.info("Stage 5: Report Generation")
            report_generator = ReportGenerator(config)
            report_generator.generate_comprehensive_report()
            logger.info("Report generation completed")
        
        logger.info("Face Recognition System Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
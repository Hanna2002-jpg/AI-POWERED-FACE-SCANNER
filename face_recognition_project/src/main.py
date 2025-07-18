#!/usr/bin/env python3
"""
Face Recognition Project - Main Execution Script
================================================

Complete pipeline for face recognition model evaluation including preprocessing,
baseline evaluation, fine-tuning, and comprehensive reporting.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

from .utils import (setup_logging, load_config, create_directories, 
                   set_random_seeds, print_system_info, save_metrics)
from .data_preprocessing import DataPreprocessor
from .baseline_evaluation import BaselineEvaluator
from .fine_tuning import FineTuner
from .final_evaluation import FinalEvaluator
from .report_generator import ReportGenerator


class FaceRecognitionPipeline:
    """Main pipeline orchestrator for face recognition evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging()
        self.start_time = time.time()
        
        # Initialize pipeline components
        self.preprocessor = DataPreprocessor(config)
        self.baseline_evaluator = BaselineEvaluator(config)
        self.fine_tuner = FineTuner(config)
        self.final_evaluator = FinalEvaluator(config)
        self.report_generator = ReportGenerator(config)
        
        # Results storage
        self.results = {
            'pipeline_start_time': self.start_time,
            'config': config,
            'stages_completed': [],
            'stage_results': {}
        }
    
    def run_complete_pipeline(self, raw_data_path: str) -> Dict[str, Any]:
        """Run the complete evaluation pipeline."""
        self.logger.info("Starting Face Recognition Evaluation Pipeline")
        self.logger.info("=" * 60)
        
        # Print system information
        print_system_info()
        
        try:
            # Stage 1: Data Preprocessing
            self._run_preprocessing_stage(raw_data_path)
            
            # Stage 2: Baseline Evaluation
            self._run_baseline_stage()
            
            # Stage 3: Model Fine-tuning
            self._run_training_stage()
            
            # Stage 4: Final Evaluation
            self._run_evaluation_stage()
            
            # Stage 5: Report Generation
            self._run_reporting_stage()
            
            # Pipeline completion
            self._finalize_pipeline()
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.results['pipeline_status'] = 'failed'
            self.results['error'] = str(e)
            raise
        
        return self.results
    
    def _run_preprocessing_stage(self, raw_data_path: str) -> None:
        """Run data preprocessing stage."""
        self.logger.info("Stage 1: Data Preprocessing")
        self.logger.info("-" * 30)
        
        stage_start = time.time()
        
        # Update config with raw data path
        self.config['dataset']['raw_path'] = raw_data_path
        
        # Run preprocessing
        self.preprocessor.process_dataset(raw_data_path)
        
        # Verify integrity
        integrity_check = self.preprocessor.verify_preprocessing_integrity()
        
        stage_time = time.time() - stage_start
        
        stage_results = {
            'status': 'completed',
            'duration': stage_time,
            'integrity_verified': integrity_check
        }
        
        self.results['stages_completed'].append('preprocessing')
        self.results['stage_results']['preprocessing'] = stage_results
        
        self.logger.info(f"Preprocessing completed in {stage_time:.2f} seconds")
        
        if not integrity_check:
            raise RuntimeError("Data integrity verification failed")
    
    def _run_baseline_stage(self) -> None:
        """Run baseline evaluation stage."""
        self.logger.info("Stage 2: Baseline Evaluation")
        self.logger.info("-" * 30)
        
        stage_start = time.time()
        
        # Run baseline evaluation
        baseline_results = self.baseline_evaluator.evaluate_all_models()
        
        stage_time = time.time() - stage_start
        
        stage_results = {
            'status': 'completed',
            'duration': stage_time,
            'model_results': baseline_results
        }
        
        self.results['stages_completed'].append('baseline')
        self.results['stage_results']['baseline'] = stage_results
        
        self.logger.info(f"Baseline evaluation completed in {stage_time:.2f} seconds")
        
        # Log key metrics
        for model_name, model_results in baseline_results.items():
            if 'error' not in model_results:
                cosine_metrics = model_results['metrics'].get('cosine', {})
                self.logger.info(f"{model_name} - ROC AUC: {cosine_metrics.get('roc_auc', 'N/A'):.4f}")
    
    def _run_training_stage(self) -> None:
        """Run model fine-tuning stage."""
        self.logger.info("Stage 3: Model Fine-tuning")
        self.logger.info("-" * 30)
        
        stage_start = time.time()
        
        # Run fine-tuning
        training_results = self.fine_tuner.train_model()
        
        stage_time = time.time() - stage_start
        
        stage_results = {
            'status': 'completed',
            'duration': stage_time,
            'training_results': training_results
        }
        
        self.results['stages_completed'].append('training')
        self.results['stage_results']['training'] = stage_results
        
        self.logger.info(f"Fine-tuning completed in {stage_time:.2f} seconds")
        
        # Log training metrics
        if 'best_val_acc' in training_results:
            self.logger.info(f"Best validation accuracy: {training_results['best_val_acc']:.4f}")
    
    def _run_evaluation_stage(self) -> None:
        """Run final evaluation stage."""
        self.logger.info("Stage 4: Final Evaluation")
        self.logger.info("-" * 30)
        
        stage_start = time.time()
        
        # Run final evaluation
        evaluation_results = self.final_evaluator.evaluate_trained_model()
        
        stage_time = time.time() - stage_start
        
        stage_results = {
            'status': 'completed',
            'duration': stage_time,
            'evaluation_results': evaluation_results
        }
        
        self.results['stages_completed'].append('evaluation')
        self.results['stage_results']['evaluation'] = stage_results
        
        self.logger.info(f"Final evaluation completed in {stage_time:.2f} seconds")
        
        # Log improvement metrics
        if 'improvement' in evaluation_results:
            improvement = evaluation_results['improvement']
            self.logger.info(f"Performance improvement: {improvement.get('accuracy_gain', 'N/A'):.4f}")
    
    def _run_reporting_stage(self) -> None:
        """Run report generation stage."""
        self.logger.info("Stage 5: Report Generation")
        self.logger.info("-" * 30)
        
        stage_start = time.time()
        
        # Generate comprehensive report
        report_results = self.report_generator.generate_report(self.results)
        
        stage_time = time.time() - stage_start
        
        stage_results = {
            'status': 'completed',
            'duration': stage_time,
            'report_files': report_results
        }
        
        self.results['stages_completed'].append('reporting')
        self.results['stage_results']['reporting'] = stage_results
        
        self.logger.info(f"Report generation completed in {stage_time:.2f} seconds")
        
        # Log report files
        for report_type, file_path in report_results.items():
            self.logger.info(f"Generated {report_type}: {file_path}")
    
    def _finalize_pipeline(self) -> None:
        """Finalize pipeline execution."""
        total_time = time.time() - self.start_time
        
        self.results['pipeline_status'] = 'completed'
        self.results['total_duration'] = total_time
        self.results['pipeline_end_time'] = time.time()
        
        # Save final results
        save_metrics(self.results, 'pipeline_results.json', self.config)
        
        self.logger.info("=" * 60)
        self.logger.info("Face Recognition Pipeline Completed Successfully!")
        self.logger.info(f"Total execution time: {total_time:.2f} seconds")
        self.logger.info(f"Stages completed: {', '.join(self.results['stages_completed'])}")
        
        # Print summary
        self._print_pipeline_summary()
    
    def _print_pipeline_summary(self) -> None:
        """Print pipeline execution summary."""
        print("\n" + "=" * 60)
        print("FACE RECOGNITION EVALUATION SUMMARY")
        print("=" * 60)
        
        # Configuration summary
        print(f"Model Architecture: {self.config['model']['architecture']}")
        print(f"Backbone: {self.config['model']['backbone']}")
        print(f"Embedding Size: {self.config['model']['embedding_size']}")
        print(f"Training Epochs: {self.config['training']['epochs']}")
        
        # Dataset summary
        print(f"\nDataset Configuration:")
        print(f"Train Split: {self.config['dataset']['train_split']}")
        print(f"Eval Split: {self.config['dataset']['eval_split']}")
        print(f"Image Size: {self.config['dataset']['image_size']}")
        
        # Results summary
        if 'baseline' in self.results['stage_results']:
            baseline_results = self.results['stage_results']['baseline']['model_results']
            model_name = self.config['model']['architecture']
            if model_name in baseline_results and 'error' not in baseline_results[model_name]:
                cosine_metrics = baseline_results[model_name]['metrics'].get('cosine', {})
                print(f"\nBaseline Performance:")
                print(f"ROC AUC: {cosine_metrics.get('roc_auc', 'N/A'):.4f}")
                print(f"Accuracy: {cosine_metrics.get('accuracy', 'N/A'):.4f}")
                print(f"EER: {cosine_metrics.get('eer', 'N/A'):.4f}")
        
        if 'evaluation' in self.results['stage_results']:
            eval_results = self.results['stage_results']['evaluation']['evaluation_results']
            if 'final_metrics' in eval_results:
                final_metrics = eval_results['final_metrics'].get('cosine', {})
                print(f"\nFinal Performance:")
                print(f"ROC AUC: {final_metrics.get('roc_auc', 'N/A'):.4f}")
                print(f"Accuracy: {final_metrics.get('accuracy', 'N/A'):.4f}")
                print(f"EER: {final_metrics.get('eer', 'N/A'):.4f}")
            
            if 'improvement' in eval_results:
                improvement = eval_results['improvement']
                print(f"\nPerformance Improvement:")
                print(f"Accuracy Gain: {improvement.get('accuracy_gain', 'N/A'):.4f}")
                print(f"ROC AUC Gain: {improvement.get('roc_auc_gain', 'N/A'):.4f}")
        
        # Execution time summary
        print(f"\nExecution Time Summary:")
        for stage, stage_results in self.results['stage_results'].items():
            duration = stage_results.get('duration', 0)
            print(f"{stage.capitalize()}: {duration:.2f} seconds")
        
        print(f"Total: {self.results.get('total_duration', 0):.2f} seconds")
        
        # Data integrity
        preprocessing_results = self.results['stage_results'].get('preprocessing', {})
        integrity_verified = preprocessing_results.get('integrity_verified', False)
        print(f"\nData Integrity: {'✓ VERIFIED' if integrity_verified else '✗ FAILED'}")
        
        print("=" * 60)


def main():
    """Main entry point for the face recognition pipeline."""
    parser = argparse.ArgumentParser(
        description="Face Recognition Model Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python -m src.main --raw-data data/raw
  
  # Run with custom config
  python -m src.main --raw-data data/raw --config custom_config.yaml
  
  # Run specific stages
  python -m src.main --raw-data data/raw --stages preprocessing baseline
        """
    )
    
    parser.add_argument(
        '--raw-data',
        type=str,
        required=True,
        help='Path to raw dataset directory'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )
    
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['preprocessing', 'baseline', 'training', 'evaluation', 'reporting', 'all'],
        default=['all'],
        help='Pipeline stages to run (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if not Path(args.config).exists():
            print(f"Error: Configuration file '{args.config}' not found")
            sys.exit(1)
        
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.output_dir:
            config['paths']['results'] = args.output_dir
            # Update other paths relative to new output dir
            for path_key in ['logs', 'plots', 'metrics']:
                config['paths'][path_key] = f"{args.output_dir}/{path_key}"
        
        # Validate raw data path
        if not Path(args.raw_data).exists():
            print(f"Error: Raw data directory '{args.raw_data}' not found")
            sys.exit(1)
        
        # Setup environment
        set_random_seeds(args.seed)
        create_directories(config)
        
        # Initialize pipeline
        pipeline = FaceRecognitionPipeline(config)
        
        # Run pipeline
        if 'all' in args.stages:
            results = pipeline.run_complete_pipeline(args.raw_data)
        else:
            # Run individual stages (implementation would go here)
            print("Individual stage execution not implemented in this example")
            sys.exit(1)
        
        print(f"\nPipeline completed successfully!")
        print(f"Results saved to: {config['paths']['results']}")
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
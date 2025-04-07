"""
Orchestration module for the Data Warehouse Subsampling Framework.

This module provides components for orchestrating the entire data processing
pipeline, including parallel execution, error handling, and reporting.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import json
import time
import datetime
import uuid
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import signal
import sys

from ..common.base import Component, ConfigManager, Pipeline, PipelineStep, ProcessingError
from ..common.utils import save_dataframe, load_dataframe, save_json, load_json, ensure_directory
from ..data_classification.data_classification import DataClassificationPipeline
from ..anomaly_detection.anomaly_detection import AnomalyDetectionPipeline
from ..core_sampling.core_sampling import SamplingPipeline
from ..data_integration.data_integration import DataIntegrationPipeline
from ..test_env_provisioning.test_env_provisioning import TestEnvironmentProvisioningPipeline

logger = logging.getLogger(__name__)


class CheckpointManager(Component):
    """Component for managing checkpoints."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the checkpoint manager.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.checkpoint_dir = None
        self.checkpoint_interval = None
        self.enabled = None
    
    def initialize(self) -> None:
        """Initialize the checkpoint manager.
        
        Raises:
            ConfigurationError: If the manager cannot be initialized.
        """
        self.enabled = self.config_manager.get('orchestration.checkpointing.enabled', True)
        self.checkpoint_dir = self.config_manager.get('orchestration.checkpointing.directory', 'checkpoints')
        self.checkpoint_interval = self.config_manager.get('orchestration.checkpointing.interval', 600)  # 10 minutes
        
        if self.enabled:
            ensure_directory(self.checkpoint_dir)
        
        self.logger.info(f"Checkpoint manager initialized with enabled={self.enabled}, directory={self.checkpoint_dir}, interval={self.checkpoint_interval}")
    
    def save_checkpoint(self, data: Dict[str, Any], stage: str) -> str:
        """Save a checkpoint.
        
        Args:
            data: Data to checkpoint.
            stage: Pipeline stage.
        
        Returns:
            Checkpoint ID.
        """
        if not self.enabled:
            return None
        
        try:
            # Generate checkpoint ID
            checkpoint_id = str(uuid.uuid4())
            
            # Create checkpoint metadata
            metadata = {
                'id': checkpoint_id,
                'stage': stage,
                'timestamp': datetime.datetime.now().isoformat(),
                'data_keys': list(data.keys())
            }
            
            # Create checkpoint directory
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id)
            ensure_directory(checkpoint_path)
            
            # Save metadata
            metadata_path = os.path.join(checkpoint_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save data
            data_path = os.path.join(checkpoint_path, 'data.json')
            
            # Filter out non-serializable data
            serializable_data = {}
            for key, value in data.items():
                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    serializable_data[key] = value
                elif isinstance(value, pd.DataFrame):
                    # Save DataFrame to CSV
                    df_path = os.path.join(checkpoint_path, f"{key}.csv")
                    value.to_csv(df_path, index=False)
                    serializable_data[key] = f"DataFrame saved to {df_path}"
                else:
                    serializable_data[key] = f"Non-serializable data of type {type(value)}"
            
            with open(data_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            self.logger.info(f"Saved checkpoint {checkpoint_id} for stage {stage}")
            
            return checkpoint_id
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
            return None
    
    def load_checkpoint(self, checkpoint_id: str) -> Tuple[Dict[str, Any], str]:
        """Load a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID.
        
        Returns:
            Tuple of (data, stage).
        """
        if not self.enabled:
            return None, None
        
        try:
            # Get checkpoint directory
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id)
            
            if not os.path.exists(checkpoint_path):
                self.logger.warning(f"Checkpoint {checkpoint_id} not found")
                return None, None
            
            # Load metadata
            metadata_path = os.path.join(checkpoint_path, 'metadata.json')
            
            if not os.path.exists(metadata_path):
                self.logger.warning(f"Metadata file not found for checkpoint {checkpoint_id}")
                return None, None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load data
            data_path = os.path.join(checkpoint_path, 'data.json')
            
            if not os.path.exists(data_path):
                self.logger.warning(f"Data file not found for checkpoint {checkpoint_id}")
                return None, None
            
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Load DataFrames
            for key in metadata.get('data_keys', []):
                df_path = os.path.join(checkpoint_path, f"{key}.csv")
                
                if os.path.exists(df_path):
                    data[key] = pd.read_csv(df_path)
            
            self.logger.info(f"Loaded checkpoint {checkpoint_id} for stage {metadata.get('stage')}")
            
            return data, metadata.get('stage')
        except Exception as e:
            self.logger.error(f"Error loading checkpoint {checkpoint_id}: {str(e)}")
            return None, None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints.
        
        Returns:
            List of checkpoint metadata.
        """
        if not self.enabled:
            return []
        
        try:
            checkpoints = []
            
            # Get checkpoint directories
            for checkpoint_id in os.listdir(self.checkpoint_dir):
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id)
                
                if not os.path.isdir(checkpoint_path):
                    continue
                
                # Load metadata
                metadata_path = os.path.join(checkpoint_path, 'metadata.json')
                
                if not os.path.exists(metadata_path):
                    continue
                
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                checkpoints.append(metadata)
            
            # Sort by timestamp
            checkpoints.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return checkpoints
        except Exception as e:
            self.logger.error(f"Error listing checkpoints: {str(e)}")
            return []
    
    def get_latest_checkpoint(self, stage: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Get the latest checkpoint.
        
        Args:
            stage: Optional pipeline stage.
        
        Returns:
            Tuple of (checkpoint ID, metadata).
        """
        if not self.enabled:
            return None, None
        
        try:
            checkpoints = self.list_checkpoints()
            
            if not checkpoints:
                return None, None
            
            # Filter by stage if provided
            if stage:
                stage_checkpoints = [c for c in checkpoints if c.get('stage') == stage]
                
                if not stage_checkpoints:
                    return None, None
                
                return stage_checkpoints[0].get('id'), stage_checkpoints[0]
            
            # Return latest checkpoint
            return checkpoints[0].get('id'), checkpoints[0]
        except Exception as e:
            self.logger.error(f"Error getting latest checkpoint: {str(e)}")
            return None, None
    
    def clean_checkpoints(self, max_age_days: int = 7, max_count: int = 100) -> int:
        """Clean old checkpoints.
        
        Args:
            max_age_days: Maximum age in days.
            max_count: Maximum number of checkpoints to keep.
        
        Returns:
            Number of checkpoints removed.
        """
        if not self.enabled:
            return 0
        
        try:
            checkpoints = self.list_checkpoints()
            
            if not checkpoints:
                return 0
            
            # Calculate cutoff date
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
            cutoff_str = cutoff_date.isoformat()
            
            # Identify checkpoints to remove
            to_remove = []
            
            # Remove old checkpoints
            for checkpoint in checkpoints:
                if checkpoint.get('timestamp', '') < cutoff_str:
                    to_remove.append(checkpoint.get('id'))
            
            # Remove excess checkpoints
            if len(checkpoints) - len(to_remove) > max_count:
                # Sort by timestamp (oldest first)
                checkpoints.sort(key=lambda x: x.get('timestamp', ''))
                
                # Add oldest checkpoints to removal list
                excess_count = len(checkpoints) - len(to_remove) - max_count
                for i in range(excess_count):
                    if checkpoints[i].get('id') not in to_remove:
                        to_remove.append(checkpoints[i].get('id'))
            
            # Remove checkpoints
            for checkpoint_id in to_remove:
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id)
                
                if os.path.exists(checkpoint_path):
                    import shutil
                    shutil.rmtree(checkpoint_path)
            
            self.logger.info(f"Cleaned {len(to_remove)} checkpoints")
            
            return len(to_remove)
        except Exception as e:
            self.logger.error(f"Error cleaning checkpoints: {str(e)}")
            return 0


class ReportGenerator(Component):
    """Component for generating reports."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the report generator.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.report_dir = None
        self.formats = None
        self.enabled = None
    
    def initialize(self) -> None:
        """Initialize the report generator.
        
        Raises:
            ConfigurationError: If the generator cannot be initialized.
        """
        self.enabled = self.config_manager.get('orchestration.reporting.enabled', True)
        self.report_dir = self.config_manager.get('orchestration.reporting.directory', 'reports')
        self.formats = self.config_manager.get('orchestration.reporting.formats', ['json', 'html', 'txt'])
        
        if self.enabled:
            ensure_directory(self.report_dir)
        
        self.logger.info(f"Report generator initialized with enabled={self.enabled}, directory={self.report_dir}, formats={self.formats}")
    
    def generate_report(self, data: Dict[str, Any], report_type: str) -> Dict[str, str]:
        """Generate a report.
        
        Args:
            data: Report data.
            report_type: Type of report.
        
        Returns:
            Dictionary mapping format to file path.
        """
        if not self.enabled:
            return {}
        
        try:
            # Generate report ID
            report_id = str(uuid.uuid4())
            
            # Create report directory
            report_path = os.path.join(self.report_dir, f"{report_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            ensure_directory(report_path)
            
            # Generate reports in different formats
            report_files = {}
            
            if 'json' in self.formats:
                json_path = os.path.join(report_path, f"{report_type}.json")
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                report_files['json'] = json_path
            
            if 'txt' in self.formats:
                txt_path = os.path.join(report_path, f"{report_type}.txt")
                with open(txt_path, 'w') as f:
                    f.write(f"Report: {report_type}\n")
                    f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
                    
                    for key, value in data.items():
                        f.write(f"{key}:\n")
                        f.write(f"{value}\n\n")
                
                report_files['txt'] = txt_path
            
            if 'html' in self.formats:
                html_path = os.path.join(report_path, f"{report_type}.html")
                with open(html_path, 'w') as f:
                    f.write(f"<html><head><title>Report: {report_type}</title></head><body>")
                    f.write(f"<h1>Report: {report_type}</h1>")
                    f.write(f"<p>Generated: {datetime.datetime.now().isoformat()}</p>")
                    
                    for key, value in data.items():
                        f.write(f"<h2>{key}</h2>")
                        f.write(f"<pre>{value}</pre>")
                    
                    f.write("</body></html>")
                
                report_files['html'] = html_path
            
            self.logger.info(f"Generated {report_type} report in formats: {list(report_files.keys())}")
            
            return report_files
        except Exception as e:
            self.logger.error(f"Error generating {report_type} report: {str(e)}")
            return {}
    
    def generate_summary_report(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate a summary report.
        
        Args:
            data: Report data.
        
        Returns:
            Dictionary mapping format to file path.
        """
        return self.generate_report(data, 'summary')
    
    def generate_error_report(self, error: Exception, stage: str, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate an error report.
        
        Args:
            error: Exception that occurred.
            stage: Pipeline stage where the error occurred.
            data: Data at the time of the error.
        
        Returns:
            Dictionary mapping format to file path.
        """
        error_data = {
            'error': str(error),
            'traceback': traceback.format_exc(),
            'stage': stage,
            'data_keys': list(data.keys())
        }
        
        return self.generate_report(error_data, 'error')


class ParallelExecutor(Component):
    """Component for parallel execution."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the parallel executor.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.enabled = None
        self.max_workers = None
        self.executor_type = None
    
    def initialize(self) -> None:
        """Initialize the parallel executor.
        
        Raises:
            ConfigurationError: If the executor cannot be initialized.
        """
        self.enabled = self.config_manager.get('orchestration.parallel.enabled', False)
        self.max_workers = self.config_manager.get('orchestration.parallel.max_workers', None)
        self.executor_type = self.config_manager.get('orchestration.parallel.executor_type', 'process')
        
        self.logger.info(f"Parallel executor initialized with enabled={self.enabled}, max_workers={self.max_workers}, executor_type={self.executor_type}")
    
    def execute(self, functions: List[Tuple[Callable, List, Dict]]) -> List[Any]:
        """Execute functions in parallel.
        
        Args:
            functions: List of (function, args, kwargs) tuples.
        
        Returns:
            List of results.
        """
        if not self.enabled or not functions:
            # Execute sequentially
            return [func(*args, **kwargs) for func, args, kwargs in functions]
        
        try:
            # Choose executor type
            if self.executor_type == 'thread':
                executor_class = ThreadPoolExecutor
            else:
                executor_class = ProcessPoolExecutor
            
            # Execute in parallel
            with executor_class(max_workers=self.max_workers) as executor:
                futures = [executor.submit(func, *args, **kwargs) for func, args, kwargs in functions]
                results = [future.result() for future in futures]
            
            return results
        except Exception as e:
            self.logger.error(f"Error in parallel execution: {str(e)}")
            
            # Fall back to sequential execution
            self.logger.info("Falling back to sequential execution")
            return [func(*args, **kwargs) for func, args, kwargs in functions]


class Orchestrator(Component):
    """Component for orchestrating the entire pipeline."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the orchestrator.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.checkpoint_manager = None
        self.report_generator = None
        self.parallel_executor = None
        self.pipelines = {}
        self.resume_from_checkpoint = None
        self.checkpoint_id = None
    
    def initialize(self) -> None:
        """Initialize the orchestrator.
        
        Raises:
            ConfigurationError: If the orchestrator cannot be initialized.
        """
        # Initialize components
        self.checkpoint_manager = CheckpointManager(self.config_manager)
        self.checkpoint_manager.initialize()
        
        self.report_generator = ReportGenerator(self.config_manager)
        self.report_generator.initialize()
        
        self.parallel_executor = ParallelExecutor(self.config_manager)
        self.parallel_executor.initialize()
        
        # Initialize pipelines
        self.pipelines = {
            'data_classification': DataClassificationPipeline(self.config_manager),
            'anomaly_detection': AnomalyDetectionPipeline(self.config_manager),
            'sampling': SamplingPipeline(self.config_manager),
            'data_integration': DataIntegrationPipeline(self.config_manager),
            'test_env_provisioning': TestEnvironmentProvisioningPipeline(self.config_manager)
        }
        
        for name, pipeline in self.pipelines.items():
            pipeline.initialize()
        
        # Get resume configuration
        self.resume_from_checkpoint = self.config_manager.get('orchestration.resume_from_checkpoint', False)
        self.checkpoint_id = self.config_manager.get('orchestration.checkpoint_id', None)
        
        self.logger.info(f"Orchestrator initialized with {len(self.pipelines)} pipelines")
    
    def run(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the entire pipeline.
        
        Args:
            input_data: Optional input data.
        
        Returns:
            Pipeline results.
        """
        self.logger.info("Starting pipeline execution")
        
        # Initialize data
        data = input_data or {}
        
        # Resume from checkpoint if configured
        if self.resume_from_checkpoint:
            if self.checkpoint_id:
                self.logger.info(f"Resuming from checkpoint {self.checkpoint_id}")
                checkpoint_data, stage = self.checkpoint_manager.load_checkpoint(self.checkpoint_id)
                
                if checkpoint_data:
                    data = checkpoint_data
                    self.logger.info(f"Resumed from checkpoint {self.checkpoint_id} at stage {stage}")
                else:
                    self.logger.warning(f"Failed to load checkpoint {self.checkpoint_id}")
            else:
                # Get latest checkpoint
                checkpoint_id, metadata = self.checkpoint_manager.get_latest_checkpoint()
                
                if checkpoint_id:
                    self.logger.info(f"Resuming from latest checkpoint {checkpoint_id}")
                    checkpoint_data, stage = self.checkpoint_manager.load_checkpoint(checkpoint_id)
                    
                    if checkpoint_data:
                        data = checkpoint_data
                        self.logger.info(f"Resumed from checkpoint {checkpoint_id} at stage {stage}")
                    else:
                        self.logger.warning(f"Failed to load checkpoint {checkpoint_id}")
                else:
                    self.logger.info("No checkpoints found, starting from scratch")
        
        try:
            # Run data classification pipeline
            self.logger.info("Running data classification pipeline")
            data = self.pipelines['data_classification'].run(data)
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(data, 'data_classification')
            
            # Run anomaly detection pipeline
            self.logger.info("Running anomaly detection pipeline")
            data = self.pipelines['anomaly_detection'].run(data)
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(data, 'anomaly_detection')
            
            # Run sampling pipeline
            self.logger.info("Running sampling pipeline")
            data = self.pipelines['sampling'].run(data)
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(data, 'sampling')
            
            # Run data integration pipeline
            self.logger.info("Running data integration pipeline")
            data = self.pipelines['data_integration'].run(data)
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(data, 'data_integration')
            
            # Run test environment provisioning pipeline
            self.logger.info("Running test environment provisioning pipeline")
            data = self.pipelines['test_env_provisioning'].run(data)
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(data, 'test_env_provisioning')
            
            # Generate summary report
            summary_data = {
                'pipeline_status': 'completed',
                'execution_time': str(datetime.datetime.now()),
                'data_classification_results': data.get('data_classification_results', {}),
                'anomaly_detection_results': data.get('anomaly_detection_results', {}),
                'sampling_results': data.get('sampling_results', {}),
                'integration_results': data.get('integration_results', {}),
                'provisioning_results': data.get('provisioning_results', {})
            }
            
            report_files = self.report_generator.generate_summary_report(summary_data)
            
            # Add report files to data
            data['report_files'] = report_files
            
            self.logger.info("Pipeline execution completed successfully")
            
            return data
        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {str(e)}")
            
            # Generate error report
            error_files = self.report_generator.generate_error_report(e, 'orchestration', data)
            
            # Add error files to data
            data['error_files'] = error_files
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(data, 'error')
            
            raise
    
    def run_pipeline(self, pipeline_name: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a specific pipeline.
        
        Args:
            pipeline_name: Name of the pipeline to run.
            data: Optional input data.
        
        Returns:
            Pipeline results.
        """
        if pipeline_name not in self.pipelines:
            self.logger.error(f"Pipeline {pipeline_name} not found")
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        self.logger.info(f"Running pipeline {pipeline_name}")
        
        try:
            # Run pipeline
            result = self.pipelines[pipeline_name].run(data or {})
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(result, pipeline_name)
            
            return result
        except Exception as e:
            self.logger.error(f"Error in pipeline {pipeline_name}: {str(e)}")
            
            # Generate error report
            error_files = self.report_generator.generate_error_report(e, pipeline_name, data or {})
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(data or {}, f"error_{pipeline_name}")
            
            raise
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up resources")
        
        try:
            # Clean old checkpoints
            self.checkpoint_manager.clean_checkpoints()
            
            # Clean up pipelines
            for name, pipeline in self.pipelines.items():
                if hasattr(pipeline, 'cleanup'):
                    pipeline.cleanup()
        except Exception as e:
            self.logger.error(f"Error cleaning up resources: {str(e)}")


class OrchestratorCLI:
    """Command-line interface for the orchestrator."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.config_manager = None
        self.orchestrator = None
    
    def parse_args(self) -> Dict[str, Any]:
        """Parse command-line arguments.
        
        Returns:
            Dictionary of arguments.
        """
        import argparse
        
        parser = argparse.ArgumentParser(description='Data Warehouse Subsampling Framework')
        
        parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
        parser.add_argument('--output-dir', type=str, help='Output directory')
        parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
        parser.add_argument('--checkpoint-id', type=str, help='Checkpoint ID to resume from')
        parser.add_argument('--pipeline', type=str, help='Run a specific pipeline')
        parser.add_argument('--list-checkpoints', action='store_true', help='List checkpoints')
        parser.add_argument('--clean-checkpoints', action='store_true', help='Clean old checkpoints')
        parser.add_argument('--env-type', type=str, choices=['file', 'database', 'native_process'], help='Test environment type')
        parser.add_argument('--no-docker', action='store_true', help='Disable Docker (use file-based or database environments)')
        
        args = parser.parse_args()
        
        return vars(args)
    
    def initialize(self, args: Dict[str, Any]) -> None:
        """Initialize the CLI.
        
        Args:
            args: Command-line arguments.
        """
        from ..common.base import ConfigManager
        
        # Load configuration
        config_path = args.get('config', 'config.yaml')
        
        self.config_manager = ConfigManager(config_path)
        
        # Override configuration with command-line arguments
        if args.get('output_dir'):
            self.config_manager.set('output_dir', args['output_dir'])
        
        if args.get('resume'):
            self.config_manager.set('orchestration.resume_from_checkpoint', True)
        
        if args.get('checkpoint_id'):
            self.config_manager.set('orchestration.checkpoint_id', args['checkpoint_id'])
        
        if args.get('env_type'):
            self.config_manager.set('test_env_provisioning.default_type', args['env_type'])
        
        if args.get('no_docker'):
            # Disable Docker
            self.config_manager.set('test_env_provisioning.docker.enabled', False)
            
            # Set default environment type to file if not specified
            if not args.get('env_type'):
                self.config_manager.set('test_env_provisioning.default_type', 'file')
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator(self.config_manager)
        self.orchestrator.initialize()
    
    def run(self) -> int:
        """Run the CLI.
        
        Returns:
            Exit code.
        """
        try:
            # Parse arguments
            args = self.parse_args()
            
            # Initialize
            self.initialize(args)
            
            # Handle special commands
            if args.get('list_checkpoints'):
                checkpoints = self.orchestrator.checkpoint_manager.list_checkpoints()
                
                print(f"Found {len(checkpoints)} checkpoints:")
                for checkpoint in checkpoints:
                    print(f"  {checkpoint.get('id')}: {checkpoint.get('stage')} ({checkpoint.get('timestamp')})")
                
                return 0
            
            if args.get('clean_checkpoints'):
                removed = self.orchestrator.checkpoint_manager.clean_checkpoints()
                
                print(f"Removed {removed} checkpoints")
                
                return 0
            
            # Run pipeline
            if args.get('pipeline'):
                pipeline_name = args['pipeline']
                
                print(f"Running pipeline: {pipeline_name}")
                
                result = self.orchestrator.run_pipeline(pipeline_name)
                
                print(f"Pipeline {pipeline_name} completed successfully")
                
                return 0
            
            # Run full pipeline
            print("Running full pipeline")
            
            result = self.orchestrator.run()
            
            print("Pipeline completed successfully")
            
            return 0
        except Exception as e:
            print(f"Error: {str(e)}")
            traceback.print_exc()
            return 1
        finally:
            # Clean up
            if self.orchestrator:
                self.orchestrator.cleanup()


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code.
    """
    # Set up signal handlers
    def signal_handler(sig, frame):
        print("Interrupted, exiting...")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run CLI
    cli = OrchestratorCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())

"""
Orchestration Module for the Data Warehouse Subsampling Framework.

This module provides the main orchestration and workflow for the entire framework,
tying together all the layers of the data subsampling architecture:
1. Data Classification & Partitioning Layer
2. Anomaly Detection & Isolation Layer
3. Core Sampling Layer
4. Data Integration Layer
5. Test Environment Provisioning Layer
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import yaml
import argparse
import sys
import time
import traceback

from ..common.base import Component, ConfigManager, PipelineStep, Pipeline, ProcessingError, ValidationError
from ..data_classification.data_classification import DataClassificationPipeline
from ..anomaly_detection.anomaly_detection import AnomalyDetectionPipeline
from ..core_sampling.core_sampling import CoreSamplingPipeline
from ..data_integration.data_integration import DataIntegrationPipeline
from ..test_env_provisioning.test_env_provisioning import TestEnvironmentProvisioningPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dwsf.log')
    ]
)

logger = logging.getLogger(__name__)


class WorkflowOrchestrator(Component):
    """Component for orchestrating the entire data warehouse subsampling workflow."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the workflow orchestrator.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.output_dir = None
        self.data_classification_pipeline = None
        self.anomaly_detection_pipeline = None
        self.core_sampling_pipeline = None
        self.data_integration_pipeline = None
        self.test_env_provisioning_pipeline = None
    
    def initialize(self) -> None:
        """Initialize the workflow orchestrator.
        
        Raises:
            ConfigurationError: If the orchestrator cannot be initialized.
        """
        # Create output directory
        self.output_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'orchestration'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize pipelines
        self.data_classification_pipeline = DataClassificationPipeline(self.config_manager)
        self.data_classification_pipeline.initialize()
        
        self.anomaly_detection_pipeline = AnomalyDetectionPipeline(self.config_manager)
        self.anomaly_detection_pipeline.initialize()
        
        self.core_sampling_pipeline = CoreSamplingPipeline(self.config_manager)
        self.core_sampling_pipeline.initialize()
        
        self.data_integration_pipeline = DataIntegrationPipeline(self.config_manager)
        self.data_integration_pipeline.initialize()
        
        self.test_env_provisioning_pipeline = TestEnvironmentProvisioningPipeline(self.config_manager)
        self.test_env_provisioning_pipeline.initialize()
        
        self.logger.info("Workflow orchestrator initialized")
    
    def validate(self) -> bool:
        """Validate the workflow orchestrator configuration and state.
        
        Returns:
            True if the orchestrator is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        # Validate output directory
        if not os.path.exists(self.output_dir):
            raise ValidationError(f"Output directory does not exist: {self.output_dir}")
        
        # Validate pipelines
        self.data_classification_pipeline.validate()
        self.anomaly_detection_pipeline.validate()
        self.core_sampling_pipeline.validate()
        self.data_integration_pipeline.validate()
        self.test_env_provisioning_pipeline.validate()
        
        return True
    
    def run_workflow(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the entire data warehouse subsampling workflow.
        
        Args:
            input_data: Optional initial input data.
        
        Returns:
            Dictionary with workflow results.
        
        Raises:
            ProcessingError: If workflow execution fails.
        """
        self.logger.info("Starting data warehouse subsampling workflow")
        
        # Initialize workflow data
        workflow_data = input_data or {}
        
        # Create workflow metadata
        workflow_metadata = {
            'workflow_id': f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now(),
            'status': 'running',
            'steps': []
        }
        
        # Add workflow metadata to workflow data
        workflow_data['workflow_metadata'] = workflow_metadata
        
        try:
            # Step 1: Data Classification & Partitioning
            self.logger.info("Step 1: Running Data Classification & Partitioning")
            step_start_time = datetime.now()
            
            workflow_data = self.data_classification_pipeline.execute(workflow_data)
            
            step_metadata = {
                'step': 'data_classification',
                'start_time': step_start_time,
                'end_time': datetime.now(),
                'status': 'completed'
            }
            workflow_metadata['steps'].append(step_metadata)
            
            # Save intermediate results
            self._save_intermediate_results(workflow_data, 'data_classification')
            
            # Step 2: Anomaly Detection & Isolation
            self.logger.info("Step 2: Running Anomaly Detection & Isolation")
            step_start_time = datetime.now()
            
            workflow_data = self.anomaly_detection_pipeline.execute(workflow_data)
            
            step_metadata = {
                'step': 'anomaly_detection',
                'start_time': step_start_time,
                'end_time': datetime.now(),
                'status': 'completed'
            }
            workflow_metadata['steps'].append(step_metadata)
            
            # Save intermediate results
            self._save_intermediate_results(workflow_data, 'anomaly_detection')
            
            # Step 3: Core Sampling
            self.logger.info("Step 3: Running Core Sampling")
            step_start_time = datetime.now()
            
            workflow_data = self.core_sampling_pipeline.execute(workflow_data)
            
            step_metadata = {
                'step': 'core_sampling',
                'start_time': step_start_time,
                'end_time': datetime.now(),
                'status': 'completed'
            }
            workflow_metadata['steps'].append(step_metadata)
            
            # Save intermediate results
            self._save_intermediate_results(workflow_data, 'core_sampling')
            
            # Step 4: Data Integration
            self.logger.info("Step 4: Running Data Integration")
            step_start_time = datetime.now()
            
            workflow_data = self.data_integration_pipeline.execute(workflow_data)
            
            step_metadata = {
                'step': 'data_integration',
                'start_time': step_start_time,
                'end_time': datetime.now(),
                'status': 'completed'
            }
            workflow_metadata['steps'].append(step_metadata)
            
            # Save intermediate results
            self._save_intermediate_results(workflow_data, 'data_integration')
            
            # Step 5: Test Environment Provisioning
            self.logger.info("Step 5: Running Test Environment Provisioning")
            step_start_time = datetime.now()
            
            workflow_data = self.test_env_provisioning_pipeline.execute(workflow_data)
            
            step_metadata = {
                'step': 'test_env_provisioning',
                'start_time': step_start_time,
                'end_time': datetime.now(),
                'status': 'completed'
            }
            workflow_metadata['steps'].append(step_metadata)
            
            # Update workflow metadata
            workflow_metadata['end_time'] = datetime.now()
            workflow_metadata['status'] = 'completed'
            
            # Save final results
            self._save_final_results(workflow_data)
            
            self.logger.info("Data warehouse subsampling workflow completed successfully")
            return workflow_data
        except Exception as e:
            # Update workflow metadata
            workflow_metadata['end_time'] = datetime.now()
            workflow_metadata['status'] = 'failed'
            workflow_metadata['error'] = str(e)
            
            # Save error results
            self._save_error_results(workflow_data, e)
            
            self.logger.error(f"Error in data warehouse subsampling workflow: {str(e)}")
            traceback.print_exc()
            
            raise ProcessingError(f"Error in data warehouse subsampling workflow: {str(e)}")
    
    def _save_intermediate_results(self, workflow_data: Dict[str, Any], step_name: str) -> None:
        """Save intermediate workflow results.
        
        Args:
            workflow_data: Workflow data.
            step_name: Name of the step.
        """
        # Create step directory
        step_dir = os.path.join(self.output_dir, step_name)
        os.makedirs(step_dir, exist_ok=True)
        
        # Save workflow metadata
        metadata_file = os.path.join(step_dir, 'workflow_metadata.json')
        with open(metadata_file, 'w') as f:
            # Convert datetime objects to strings
            metadata = workflow_data.get('workflow_metadata', {})
            metadata_copy = metadata.copy()
            
            if 'start_time' in metadata_copy:
                metadata_copy['start_time'] = metadata_copy['start_time'].isoformat()
            
            if 'end_time' in metadata_copy:
                metadata_copy['end_time'] = metadata_copy['end_time'].isoformat()
            
            for step in metadata_copy.get('steps', []):
                if 'start_time' in step:
                    step['start_time'] = step['start_time'].isoformat()
                
                if 'end_time' in step:
                    step['end_time'] = step['end_time'].isoformat()
            
            json.dump(metadata_copy, f, indent=2)
        
        # Save step-specific results
        if step_name == 'data_classification':
            # Save domain partitions summary
            if 'domain_partitions' in workflow_data:
                summary = {}
                for domain, tables in workflow_data['domain_partitions'].items():
                    summary[domain] = {table_name: len(df) for table_name, df in tables.items()}
                
                summary_file = os.path.join(step_dir, 'domain_partitions_summary.json')
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
            
            # Save relationships
            if 'relationships' in workflow_data:
                relationships = []
                for rel in workflow_data['relationships']:
                    relationships.append({
                        'parent_table': rel.parent_table,
                        'parent_column': rel.parent_column,
                        'child_table': rel.child_table,
                        'child_column': rel.child_column
                    })
                
                relationships_file = os.path.join(step_dir, 'relationships.json')
                with open(relationships_file, 'w') as f:
                    json.dump(relationships, f, indent=2)
        
        elif step_name == 'anomaly_detection':
            # Save anomalies summary
            if 'anomalies' in workflow_data:
                summary = {
                    'total_anomalies': len(workflow_data['anomalies']),
                    'anomalies_by_table': {}
                }
                
                for anomaly in workflow_data['anomalies']:
                    table_name = anomaly.get('table_name', 'unknown')
                    if table_name not in summary['anomalies_by_table']:
                        summary['anomalies_by_table'][table_name] = 0
                    
                    summary['anomalies_by_table'][table_name] += 1
                
                summary_file = os.path.join(step_dir, 'anomalies_summary.json')
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
        
        elif step_name == 'core_sampling':
            # Save sampling results
            if 'sampling_results' in workflow_data:
                results = []
                for result in workflow_data['sampling_results']:
                    results.append(result.to_dict())
                
                results_file = os.path.join(step_dir, 'sampling_results.json')
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
        
        elif step_name == 'data_integration':
            # Save integration results
            if 'integration_results' in workflow_data:
                results = []
                for result in workflow_data['integration_results']:
                    results.append(result.to_dict())
                
                results_file = os.path.join(step_dir, 'integration_results.json')
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
            
            # Save export paths
            if 'export_paths' in workflow_data:
                export_paths_file = os.path.join(step_dir, 'export_paths.json')
                with open(export_paths_file, 'w') as f:
                    json.dump(workflow_data['export_paths'], f, indent=2)
        
        elif step_name == 'test_env_provisioning':
            # Save environment statuses
            if 'environment_statuses' in workflow_data:
                statuses = []
                for status in workflow_data['environment_statuses']:
                    statuses.append(status.to_dict())
                
                statuses_file = os.path.join(step_dir, 'environment_statuses.json')
                with open(statuses_file, 'w') as f:
                    json.dump(statuses, f, indent=2)
    
    def _save_final_results(self, workflow_data: Dict[str, Any]) -> None:
        """Save final workflow results.
        
        Args:
            workflow_data: Workflow data.
        """
        # Create results directory
        results_dir = os.path.join(self.output_dir, 'final_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save workflow metadata
        metadata_file = os.path.join(results_dir, 'workflow_metadata.json')
        with open(metadata_file, 'w') as f:
            # Convert datetime objects to strings
            metadata = workflow_data.get('workflow_metadata', {})
            metadata_copy = metadata.copy()
            
            if 'start_time' in metadata_copy:
                metadata_copy['start_time'] = metadata_copy['start_time'].isoformat()
            
            if 'end_time' in metadata_copy:
                metadata_copy['end_time'] = metadata_copy['end_time'].isoformat()
            
            for step in metadata_copy.get('steps', []):
                if 'start_time' in step:
                    step['start_time'] = step['start_time'].isoformat()
                
                if 'end_time' in step:
                    step['end_time'] = step['end_time'].isoformat()
            
            json.dump(metadata_copy, f, indent=2)
        
        # Save summary
        summary = {
            'workflow_id': workflow_data.get('workflow_metadata', {}).get('workflow_id'),
            'start_time': workflow_data.get('workflow_metadata', {}).get('start_time').isoformat() if workflow_data.get('workflow_metadata', {}).get('start_time') else None,
            'end_time': workflow_data.get('workflow_metadata', {}).get('end_time').isoformat() if workflow_data.get('workflow_metadata', {}).get('end_time') else None,
            'status': workflow_data.get('workflow_metadata', {}).get('status'),
            'data_reduction': {}
        }
        
        # Calculate data reduction
        if 'domain_partitions' in workflow_data and 'original_data' in workflow_data:
            original_row_count = 0
            for domain, tables in workflow_data['original_data'].items():
                for table_name, df in tables.items():
                    original_row_count += len(df)
            
            final_row_count = 0
            for domain, tables in workflow_data['domain_partitions'].items():
                for table_name, df in tables.items():
                    final_row_count += len(df)
            
            if original_row_count > 0:
                reduction_ratio = (original_row_count - final_row_count) / original_row_count
                summary['data_reduction'] = {
                    'original_row_count': original_row_count,
                    'final_row_count': final_row_count,
                    'reduction_ratio': reduction_ratio,
                    'reduction_percentage': f"{reduction_ratio * 100:.2f}%"
                }
        
        # Add anomaly statistics
        if 'anomalies' in workflow_data:
            summary['anomalies'] = {
                'total_anomalies': len(workflow_data['anomalies']),
                'anomalies_by_table': {}
            }
            
            for anomaly in workflow_data['anomalies']:
                table_name = anomaly.get('table_name', 'unknown')
                if table_name not in summary['anomalies']['anomalies_by_table']:
                    summary['anomalies']['anomalies_by_table'][table_name] = 0
                
                summary['anomalies']['anomalies_by_table'][table_name] += 1
        
        # Add environment statistics
        if 'environment_statuses' in workflow_data:
            summary['environments'] = {
                'total_environments': len(workflow_data['environment_statuses']),
                'running_environments': sum(1 for status in workflow_data['environment_statuses'] if status.status == 'running'),
                'environments': []
            }
            
            for status in workflow_data['environment_statuses']:
                summary['environments']['environments'].append({
                    'name': status.name,
                    'status': status.status,
                    'connection_info': status.connection_info
                })
        
        # Save summary
        summary_file = os.path.join(results_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create a human-readable report
        report = f"""
Data Warehouse Subsampling Framework - Workflow Report
======================================================

Workflow ID: {summary.get('workflow_id')}
Start Time: {summary.get('start_time')}
End Time: {summary.get('end_time')}
Status: {summary.get('status')}

Data Reduction
-------------
Original Row Count: {summary.get('data_reduction', {}).get('original_row_count', 'N/A')}
Final Row Count: {summary.get('data_reduction', {}).get('final_row_count', 'N/A')}
Reduction Ratio: {summary.get('data_reduction', {}).get('reduction_ratio', 'N/A')}
Reduction Percentage: {summary.get('data_reduction', {}).get('reduction_percentage', 'N/A')}

Anomalies
---------
Total Anomalies: {summary.get('anomalies', {}).get('total_anomalies', 'N/A')}
Anomalies by Table:
"""
        
        if 'anomalies' in summary:
            for table_name, count in summary['anomalies']['anomalies_by_table'].items():
                report += f"  - {table_name}: {count}\n"
        
        report += """
Test Environments
----------------
"""
        
        if 'environments' in summary:
            report += f"Total Environments: {summary['environments']['total_environments']}\n"
            report += f"Running Environments: {summary['environments']['running_environments']}\n"
            report += "Environments:\n"
            
            for env in summary['environments']['environments']:
                report += f"  - {env['name']} ({env['status']})\n"
                if env['connection_info'] and 'ports' in env['connection_info']:
                    report += "    Connection Ports:\n"
                    for container_port, host_port in env['connection_info']['ports'].items():
                        report += f"      {container_port} -> {host_port}\n"
        
        # Save report
        report_file = os.path.join(results_dir, 'report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
    
    def _save_error_results(self, workflow_data: Dict[str, Any], error: Exception) -> None:
        """Save error workflow results.
        
        Args:
            workflow_data: Workflow data.
            error: Exception that occurred.
        """
        # Create error directory
        error_dir = os.path.join(self.output_dir, 'error')
        os.makedirs(error_dir, exist_ok=True)
        
        # Save workflow metadata
        metadata_file = os.path.join(error_dir, 'workflow_metadata.json')
        with open(metadata_file, 'w') as f:
            # Convert datetime objects to strings
            metadata = workflow_data.get('workflow_metadata', {})
            metadata_copy = metadata.copy()
            
            if 'start_time' in metadata_copy:
                metadata_copy['start_time'] = metadata_copy['start_time'].isoformat()
            
            if 'end_time' in metadata_copy:
                metadata_copy['end_time'] = metadata_copy['end_time'].isoformat()
            
            for step in metadata_copy.get('steps', []):
                if 'start_time' in step:
                    step['start_time'] = step['start_time'].isoformat()
                
                if 'end_time' in step:
                    step['end_time'] = step['end_time'].isoformat()
            
            json.dump(metadata_copy, f, indent=2)
        
        # Save error details
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        
        error_file = os.path.join(error_dir, 'error_details.json')
        with open(error_file, 'w') as f:
            json.dump(error_details, f, indent=2)


class ParallelWorkflowOrchestrator(WorkflowOrchestrator):
    """Component for orchestrating the data warehouse subsampling workflow with parallel processing."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the parallel workflow orchestrator.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.parallel_config = None
    
    def initialize(self) -> None:
        """Initialize the parallel workflow orchestrator.
        
        Raises:
            ConfigurationError: If the orchestrator cannot be initialized.
        """
        # Initialize parent
        super().initialize()
        
        # Get parallel configuration
        self.parallel_config = self.config_manager.get('orchestration.parallel', {})
        
        self.logger.info("Parallel workflow orchestrator initialized")
    
    def run_workflow(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the entire data warehouse subsampling workflow with parallel processing.
        
        Args:
            input_data: Optional initial input data.
        
        Returns:
            Dictionary with workflow results.
        
        Raises:
            ProcessingError: If workflow execution fails.
        """
        self.logger.info("Starting parallel data warehouse subsampling workflow")
        
        # Initialize workflow data
        workflow_data = input_data or {}
        
        # Create workflow metadata
        workflow_metadata = {
            'workflow_id': f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now(),
            'status': 'running',
            'steps': []
        }
        
        # Add workflow metadata to workflow data
        workflow_data['workflow_metadata'] = workflow_metadata
        
        try:
            # Step 1: Data Classification & Partitioning
            self.logger.info("Step 1: Running Data Classification & Partitioning")
            step_start_time = datetime.now()
            
            workflow_data = self.data_classification_pipeline.execute(workflow_data)
            
            step_metadata = {
                'step': 'data_classification',
                'start_time': step_start_time,
                'end_time': datetime.now(),
                'status': 'completed'
            }
            workflow_metadata['steps'].append(step_metadata)
            
            # Save intermediate results
            self._save_intermediate_results(workflow_data, 'data_classification')
            
            # Step 2 & 3: Run Anomaly Detection and Core Sampling in parallel
            self.logger.info("Steps 2 & 3: Running Anomaly Detection & Core Sampling in parallel")
            
            # Check if parallel processing is enabled
            if self.parallel_config.get('enabled', False):
                # Import multiprocessing
                import multiprocessing as mp
                
                # Create a manager for sharing data between processes
                manager = mp.Manager()
                shared_dict = manager.dict()
                
                # Create processes
                anomaly_process = mp.Process(
                    target=self._run_anomaly_detection,
                    args=(workflow_data, shared_dict)
                )
                
                sampling_process = mp.Process(
                    target=self._run_core_sampling,
                    args=(workflow_data, shared_dict)
                )
                
                # Start processes
                anomaly_process.start()
                sampling_process.start()
                
                # Wait for processes to complete
                anomaly_process.join()
                sampling_process.join()
                
                # Update workflow data
                if 'anomaly_detection' in shared_dict:
                    anomaly_data = shared_dict['anomaly_detection']
                    workflow_data['anomalies'] = anomaly_data.get('anomalies', [])
                    workflow_metadata['steps'].append(anomaly_data.get('step_metadata', {}))
                
                if 'core_sampling' in shared_dict:
                    sampling_data = shared_dict['core_sampling']
                    workflow_data['domain_partitions'] = sampling_data.get('domain_partitions', {})
                    workflow_data['sampling_results'] = sampling_data.get('sampling_results', [])
                    workflow_metadata['steps'].append(sampling_data.get('step_metadata', {}))
            else:
                # Run sequentially
                # Step 2: Anomaly Detection & Isolation
                self.logger.info("Step 2: Running Anomaly Detection & Isolation")
                step_start_time = datetime.now()
                
                workflow_data = self.anomaly_detection_pipeline.execute(workflow_data)
                
                step_metadata = {
                    'step': 'anomaly_detection',
                    'start_time': step_start_time,
                    'end_time': datetime.now(),
                    'status': 'completed'
                }
                workflow_metadata['steps'].append(step_metadata)
                
                # Save intermediate results
                self._save_intermediate_results(workflow_data, 'anomaly_detection')
                
                # Step 3: Core Sampling
                self.logger.info("Step 3: Running Core Sampling")
                step_start_time = datetime.now()
                
                workflow_data = self.core_sampling_pipeline.execute(workflow_data)
                
                step_metadata = {
                    'step': 'core_sampling',
                    'start_time': step_start_time,
                    'end_time': datetime.now(),
                    'status': 'completed'
                }
                workflow_metadata['steps'].append(step_metadata)
                
                # Save intermediate results
                self._save_intermediate_results(workflow_data, 'core_sampling')
            
            # Step 4: Data Integration
            self.logger.info("Step 4: Running Data Integration")
            step_start_time = datetime.now()
            
            workflow_data = self.data_integration_pipeline.execute(workflow_data)
            
            step_metadata = {
                'step': 'data_integration',
                'start_time': step_start_time,
                'end_time': datetime.now(),
                'status': 'completed'
            }
            workflow_metadata['steps'].append(step_metadata)
            
            # Save intermediate results
            self._save_intermediate_results(workflow_data, 'data_integration')
            
            # Step 5: Test Environment Provisioning
            self.logger.info("Step 5: Running Test Environment Provisioning")
            step_start_time = datetime.now()
            
            workflow_data = self.test_env_provisioning_pipeline.execute(workflow_data)
            
            step_metadata = {
                'step': 'test_env_provisioning',
                'start_time': step_start_time,
                'end_time': datetime.now(),
                'status': 'completed'
            }
            workflow_metadata['steps'].append(step_metadata)
            
            # Update workflow metadata
            workflow_metadata['end_time'] = datetime.now()
            workflow_metadata['status'] = 'completed'
            
            # Save final results
            self._save_final_results(workflow_data)
            
            self.logger.info("Parallel data warehouse subsampling workflow completed successfully")
            return workflow_data
        except Exception as e:
            # Update workflow metadata
            workflow_metadata['end_time'] = datetime.now()
            workflow_metadata['status'] = 'failed'
            workflow_metadata['error'] = str(e)
            
            # Save error results
            self._save_error_results(workflow_data, e)
            
            self.logger.error(f"Error in parallel data warehouse subsampling workflow: {str(e)}")
            traceback.print_exc()
            
            raise ProcessingError(f"Error in parallel data warehouse subsampling workflow: {str(e)}")
    
    def _run_anomaly_detection(self, workflow_data: Dict[str, Any], shared_dict: Dict[str, Any]) -> None:
        """Run anomaly detection in a separate process.
        
        Args:
            workflow_data: Workflow data.
            shared_dict: Shared dictionary for returning results.
        """
        try:
            self.logger.info("Running anomaly detection in parallel process")
            step_start_time = datetime.now()
            
            # Execute anomaly detection
            result_data = self.anomaly_detection_pipeline.execute(workflow_data)
            
            # Create step metadata
            step_metadata = {
                'step': 'anomaly_detection',
                'start_time': step_start_time,
                'end_time': datetime.now(),
                'status': 'completed'
            }
            
            # Save intermediate results
            self._save_intermediate_results(result_data, 'anomaly_detection')
            
            # Update shared dictionary
            shared_dict['anomaly_detection'] = {
                'anomalies': result_data.get('anomalies', []),
                'step_metadata': step_metadata
            }
            
            self.logger.info("Anomaly detection completed in parallel process")
        except Exception as e:
            self.logger.error(f"Error in anomaly detection parallel process: {str(e)}")
            traceback.print_exc()
            
            # Update shared dictionary with error
            shared_dict['anomaly_detection'] = {
                'error': str(e),
                'step_metadata': {
                    'step': 'anomaly_detection',
                    'start_time': step_start_time,
                    'end_time': datetime.now(),
                    'status': 'failed',
                    'error': str(e)
                }
            }
    
    def _run_core_sampling(self, workflow_data: Dict[str, Any], shared_dict: Dict[str, Any]) -> None:
        """Run core sampling in a separate process.
        
        Args:
            workflow_data: Workflow data.
            shared_dict: Shared dictionary for returning results.
        """
        try:
            self.logger.info("Running core sampling in parallel process")
            step_start_time = datetime.now()
            
            # Execute core sampling
            result_data = self.core_sampling_pipeline.execute(workflow_data)
            
            # Create step metadata
            step_metadata = {
                'step': 'core_sampling',
                'start_time': step_start_time,
                'end_time': datetime.now(),
                'status': 'completed'
            }
            
            # Save intermediate results
            self._save_intermediate_results(result_data, 'core_sampling')
            
            # Update shared dictionary
            shared_dict['core_sampling'] = {
                'domain_partitions': result_data.get('domain_partitions', {}),
                'sampling_results': result_data.get('sampling_results', []),
                'step_metadata': step_metadata
            }
            
            self.logger.info("Core sampling completed in parallel process")
        except Exception as e:
            self.logger.error(f"Error in core sampling parallel process: {str(e)}")
            traceback.print_exc()
            
            # Update shared dictionary with error
            shared_dict['core_sampling'] = {
                'error': str(e),
                'step_metadata': {
                    'step': 'core_sampling',
                    'start_time': step_start_time,
                    'end_time': datetime.now(),
                    'status': 'failed',
                    'error': str(e)
                }
            }


def main():
    """Main entry point for the data warehouse subsampling framework."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Data Warehouse Subsampling Framework')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_path = args.config
        if not os.path.exists(config_path):
            print(f"Configuration file not found: {config_path}")
            return 1
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override output directory if specified
        if args.output_dir:
            config['general'] = config.get('general', {})
            config['general']['output_directory'] = args.output_dir
        
        # Override parallel processing if specified
        if args.parallel:
            config['orchestration'] = config.get('orchestration', {})
            config['orchestration']['parallel'] = config['orchestration'].get('parallel', {})
            config['orchestration']['parallel']['enabled'] = True
        
        # Create configuration manager
        config_manager = ConfigManager(config)
        
        # Create orchestrator
        if args.parallel or config.get('orchestration', {}).get('parallel', {}).get('enabled', False):
            orchestrator = ParallelWorkflowOrchestrator(config_manager)
        else:
            orchestrator = WorkflowOrchestrator(config_manager)
        
        # Initialize orchestrator
        orchestrator.initialize()
        
        # Validate orchestrator
        orchestrator.validate()
        
        # Run workflow
        orchestrator.run_workflow()
        
        print("Data warehouse subsampling workflow completed successfully")
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

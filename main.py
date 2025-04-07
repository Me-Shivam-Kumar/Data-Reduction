"""
Main entry point for the Data Warehouse Subsampling Framework.

This module provides the main entry point for running the framework
and includes command-line argument parsing and configuration loading.
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime

from .common.base import ConfigManager
from .orchestration.orchestration import WorkflowOrchestrator, ParallelWorkflowOrchestrator

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
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

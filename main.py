#!/usr/bin/env python3
"""
Main entry point for the Data Warehouse Subsampling Framework.
"""

import sys
import os
import logging
from orchestration.orchestration import OrchestratorCLI

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dwsf.log')
        ]
    )

def main():
    """Main entry point."""
    # Set up logging
    setup_logging()
    
    # Add the current directory to the Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Run the CLI
    cli = OrchestratorCLI()
    return cli.run()

if __name__ == '__main__':
    sys.exit(main())

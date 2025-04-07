"""
Test Environment Provisioning module for the Data Warehouse Subsampling Framework.

This module provides components for creating test environments using
file-based and database storage options without Docker dependencies.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import sqlite3
import shutil
import datetime
import uuid
import subprocess
import time

from ..common.base import Component, ConfigManager, Pipeline, PipelineStep, EnvironmentStatus, ProcessingError
from ..common.utils import save_dataframe, load_dataframe, save_json, load_json, ensure_directory, create_sqlite_database

logger = logging.getLogger(__name__)


class EnvironmentProvisioner(Component):
    """Base class for environment provisioners."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the environment provisioner.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
    
    def provision(self, name: str, datasets: List[str], dataset_data: Dict[str, Dict[str, pd.DataFrame]]) -> EnvironmentStatus:
        """Provision a test environment.
        
        Args:
            name: Name of the environment.
            datasets: List of dataset names to include.
            dataset_data: Dictionary mapping dataset names to dictionaries mapping table names to DataFrames.
        
        Returns:
            Environment status.
        """
        raise NotImplementedError("Subclasses must implement provision()")
    
    def deprovision(self, name: str) -> EnvironmentStatus:
        """Deprovision a test environment.
        
        Args:
            name: Name of the environment.
        
        Returns:
            Environment status.
        """
        raise NotImplementedError("Subclasses must implement deprovision()")
    
    def get_status(self, name: str) -> EnvironmentStatus:
        """Get the status of a test environment.
        
        Args:
            name: Name of the environment.
        
        Returns:
            Environment status.
        """
        raise NotImplementedError("Subclasses must implement get_status()")


class FileEnvironmentProvisioner(EnvironmentProvisioner):
    """Provisioner for file-based test environments."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the file environment provisioner.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.base_dir = None
        self.file_format = None
    
    def initialize(self) -> None:
        """Initialize the file environment provisioner.
        
        Raises:
            ConfigurationError: If the provisioner cannot be initialized.
        """
        self.base_dir = self.config_manager.get('test_env_provisioning.file.base_dir', 'environments/file')
        self.file_format = self.config_manager.get('test_env_provisioning.file.format', 'csv')
        
        # Create base directory
        ensure_directory(self.base_dir)
        
        self.logger.info(f"File environment provisioner initialized with base directory: {self.base_dir}")
    
    def provision(self, name: str, datasets: List[str], dataset_data: Dict[str, Dict[str, pd.DataFrame]]) -> EnvironmentStatus:
        """Provision a file-based test environment.
        
        Args:
            name: Name of the environment.
            datasets: List of dataset names to include.
            dataset_data: Dictionary mapping dataset names to dictionaries mapping table names to DataFrames.
        
        Returns:
            Environment status.
        """
        self.logger.info(f"Provisioning file environment: {name}")
        
        try:
            # Create environment directory
            env_dir = os.path.join(self.base_dir, name)
            ensure_directory(env_dir)
            
            # Save datasets
            dataset_files = {}
            
            for dataset_name in datasets:
                if dataset_name not in dataset_data:
                    self.logger.warning(f"Dataset {dataset_name} not found, skipping")
                    continue
                
                # Create dataset directory
                dataset_dir = os.path.join(env_dir, dataset_name)
                ensure_directory(dataset_dir)
                
                # Save tables
                table_files = {}
                
                for table_name, df in dataset_data[dataset_name].items():
                    if self.file_format == 'csv':
                        file_path = os.path.join(dataset_dir, f"{table_name}.csv")
                        df.to_csv(file_path, index=False)
                    elif self.file_format == 'parquet':
                        file_path = os.path.join(dataset_dir, f"{table_name}.parquet")
                        df.to_parquet(file_path, index=False)
                    elif self.file_format == 'json':
                        file_path = os.path.join(dataset_dir, f"{table_name}.json")
                        df.to_json(file_path, orient='records', lines=True)
                    else:
                        file_path = os.path.join(dataset_dir, f"{table_name}.csv")
                        df.to_csv(file_path, index=False)
                    
                    table_files[table_name] = file_path
                
                dataset_files[dataset_name] = table_files
            
            # Create metadata file
            metadata = {
                'name': name,
                'type': 'file',
                'datasets': datasets,
                'created_at': datetime.datetime.now().isoformat(),
                'status': 'running',
                'dataset_files': dataset_files
            }
            
            metadata_path = os.path.join(env_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Return environment status
            return EnvironmentStatus(
                name=name,
                status='running',
                connection_info={
                    'type': 'file',
                    'base_dir': env_dir,
                    'datasets': datasets,
                    'dataset_files': dataset_files
                }
            )
        except Exception as e:
            self.logger.error(f"Error provisioning file environment {name}: {str(e)}")
            return EnvironmentStatus(
                name=name,
                status='failed',
                error=str(e)
            )
    
    def deprovision(self, name: str) -> EnvironmentStatus:
        """Deprovision a file-based test environment.
        
        Args:
            name: Name of the environment.
        
        Returns:
            Environment status.
        """
        self.logger.info(f"Deprovisioning file environment: {name}")
        
        try:
            # Get environment directory
            env_dir = os.path.join(self.base_dir, name)
            
            if not os.path.exists(env_dir):
                self.logger.warning(f"Environment directory not found: {env_dir}")
                return EnvironmentStatus(
                    name=name,
                    status='stopped',
                    error=f"Environment directory not found: {env_dir}"
                )
            
            # Remove environment directory
            shutil.rmtree(env_dir)
            
            # Return environment status
            return EnvironmentStatus(
                name=name,
                status='stopped'
            )
        except Exception as e:
            self.logger.error(f"Error deprovisioning file environment {name}: {str(e)}")
            return EnvironmentStatus(
                name=name,
                status='failed',
                error=str(e)
            )
    
    def get_status(self, name: str) -> EnvironmentStatus:
        """Get the status of a file-based test environment.
        
        Args:
            name: Name of the environment.
        
        Returns:
            Environment status.
        """
        self.logger.info(f"Getting status of file environment: {name}")
        
        try:
            # Get environment directory
            env_dir = os.path.join(self.base_dir, name)
            
            if not os.path.exists(env_dir):
                self.logger.warning(f"Environment directory not found: {env_dir}")
                return EnvironmentStatus(
                    name=name,
                    status='stopped',
                    error=f"Environment directory not found: {env_dir}"
                )
            
            # Get metadata file
            metadata_path = os.path.join(env_dir, 'metadata.json')
            
            if not os.path.exists(metadata_path):
                self.logger.warning(f"Metadata file not found: {metadata_path}")
                return EnvironmentStatus(
                    name=name,
                    status='unknown',
                    error=f"Metadata file not found: {metadata_path}"
                )
            
            # Read metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Return environment status
            return EnvironmentStatus(
                name=name,
                status=metadata.get('status', 'unknown'),
                connection_info={
                    'type': 'file',
                    'base_dir': env_dir,
                    'datasets': metadata.get('datasets', []),
                    'dataset_files': metadata.get('dataset_files', {})
                }
            )
        except Exception as e:
            self.logger.error(f"Error getting status of file environment {name}: {str(e)}")
            return EnvironmentStatus(
                name=name,
                status='unknown',
                error=str(e)
            )


class DatabaseEnvironmentProvisioner(EnvironmentProvisioner):
    """Provisioner for database test environments."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the database environment provisioner.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.base_dir = None
        self.db_type = None
    
    def initialize(self) -> None:
        """Initialize the database environment provisioner.
        
        Raises:
            ConfigurationError: If the provisioner cannot be initialized.
        """
        self.base_dir = self.config_manager.get('test_env_provisioning.database.base_dir', 'environments/database')
        self.db_type = self.config_manager.get('test_env_provisioning.database.db_type', 'sqlite')
        
        # Create base directory
        ensure_directory(self.base_dir)
        
        self.logger.info(f"Database environment provisioner initialized with base directory: {self.base_dir}")
    
    def provision(self, name: str, datasets: List[str], dataset_data: Dict[str, Dict[str, pd.DataFrame]]) -> EnvironmentStatus:
        """Provision a database test environment.
        
        Args:
            name: Name of the environment.
            datasets: List of dataset names to include.
            dataset_data: Dictionary mapping dataset names to dictionaries mapping table names to DataFrames.
        
        Returns:
            Environment status.
        """
        self.logger.info(f"Provisioning database environment: {name}")
        
        try:
            # Create environment directory
            env_dir = os.path.join(self.base_dir, name)
            ensure_directory(env_dir)
            
            # Create database
            if self.db_type == 'sqlite':
                # Create SQLite database
                db_path = os.path.join(env_dir, f"{name}.db")
                
                # Combine all datasets into a single dictionary
                all_tables = {}
                for dataset_name in datasets:
                    if dataset_name not in dataset_data:
                        self.logger.warning(f"Dataset {dataset_name} not found, skipping")
                        continue
                    
                    # Add dataset prefix to table names to avoid conflicts
                    for table_name, df in dataset_data[dataset_name].items():
                        all_tables[f"{dataset_name}_{table_name}"] = df
                
                # Create database
                conn = sqlite3.connect(db_path)
                
                for table_name, df in all_tables.items():
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                
                conn.close()
                
                # Create metadata file
                metadata = {
                    'name': name,
                    'type': 'database',
                    'db_type': 'sqlite',
                    'datasets': datasets,
                    'created_at': datetime.datetime.now().isoformat(),
                    'status': 'running',
                    'db_path': db_path,
                    'tables': list(all_tables.keys())
                }
                
                metadata_path = os.path.join(env_dir, 'metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Return environment status
                return EnvironmentStatus(
                    name=name,
                    status='running',
                    connection_info={
                        'type': 'database',
                        'db_type': 'sqlite',
                        'db_path': db_path,
                        'tables': list(all_tables.keys())
                    }
                )
            elif self.db_type == 'in-memory':
                # Create in-memory database (for testing purposes)
                # Note: This is not persistent and will be lost when the process ends
                
                # Combine all datasets into a single dictionary
                all_tables = {}
                for dataset_name in datasets:
                    if dataset_name not in dataset_data:
                        self.logger.warning(f"Dataset {dataset_name} not found, skipping")
                        continue
                    
                    # Add dataset prefix to table names to avoid conflicts
                    for table_name, df in dataset_data[dataset_name].items():
                        all_tables[f"{dataset_name}_{table_name}"] = df
                
                # Create metadata file
                metadata = {
                    'name': name,
                    'type': 'database',
                    'db_type': 'in-memory',
                    'datasets': datasets,
                    'created_at': datetime.datetime.now().isoformat(),
                    'status': 'running',
                    'tables': list(all_tables.keys())
                }
                
                metadata_path = os.path.join(env_dir, 'metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Return environment status
                return EnvironmentStatus(
                    name=name,
                    status='running',
                    connection_info={
                        'type': 'database',
                        'db_type': 'in-memory',
                        'tables': list(all_tables.keys())
                    }
                )
            else:
                self.logger.error(f"Unsupported database type: {self.db_type}")
                return EnvironmentStatus(
                    name=name,
                    status='failed',
                    error=f"Unsupported database type: {self.db_type}"
                )
        except Exception as e:
            self.logger.error(f"Error provisioning database environment {name}: {str(e)}")
            return EnvironmentStatus(
                name=name,
                status='failed',
                error=str(e)
            )
    
    def deprovision(self, name: str) -> EnvironmentStatus:
        """Deprovision a database test environment.
        
        Args:
            name: Name of the environment.
        
        Returns:
            Environment status.
        """
        self.logger.info(f"Deprovisioning database environment: {name}")
        
        try:
            # Get environment directory
            env_dir = os.path.join(self.base_dir, name)
            
            if not os.path.exists(env_dir):
                self.logger.warning(f"Environment directory not found: {env_dir}")
                return EnvironmentStatus(
                    name=name,
                    status='stopped',
                    error=f"Environment directory not found: {env_dir}"
                )
            
            # Remove environment directory
            shutil.rmtree(env_dir)
            
            # Return environment status
            return EnvironmentStatus(
                name=name,
                status='stopped'
            )
        except Exception as e:
            self.logger.error(f"Error deprovisioning database environment {name}: {str(e)}")
            return EnvironmentStatus(
                name=name,
                status='failed',
                error=str(e)
            )
    
    def get_status(self, name: str) -> EnvironmentStatus:
        """Get the status of a database test environment.
        
        Args:
            name: Name of the environment.
        
        Returns:
            Environment status.
        """
        self.logger.info(f"Getting status of database environment: {name}")
        
        try:
            # Get environment directory
            env_dir = os.path.join(self.base_dir, name)
            
            if not os.path.exists(env_dir):
                self.logger.warning(f"Environment directory not found: {env_dir}")
                return EnvironmentStatus(
                    name=name,
                    status='stopped',
                    error=f"Environment directory not found: {env_dir}"
                )
            
            # Get metadata file
            metadata_path = os.path.join(env_dir, 'metadata.json')
            
            if not os.path.exists(metadata_path):
                self.logger.warning(f"Metadata file not found: {metadata_path}")
                return EnvironmentStatus(
                    name=name,
                    status='unknown',
                    error=f"Metadata file not found: {metadata_path}"
                )
            
            # Read metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check database
            if metadata.get('db_type') == 'sqlite':
                db_path = metadata.get('db_path')
                
                if not os.path.exists(db_path):
                    self.logger.warning(f"Database file not found: {db_path}")
                    return EnvironmentStatus(
                        name=name,
                        status='failed',
                        error=f"Database file not found: {db_path}"
                    )
                
                # Test database connection
                try:
                    conn = sqlite3.connect(db_path)
                    conn.close()
                except Exception as e:
                    self.logger.warning(f"Error connecting to database: {str(e)}")
                    return EnvironmentStatus(
                        name=name,
                        status='failed',
                        error=f"Error connecting to database: {str(e)}"
                    )
            
            # Return environment status
            return EnvironmentStatus(
                name=name,
                status=metadata.get('status', 'unknown'),
                connection_info={
                    'type': 'database',
                    'db_type': metadata.get('db_type'),
                    'db_path': metadata.get('db_path') if metadata.get('db_type') == 'sqlite' else None,
                    'tables': metadata.get('tables', [])
                }
            )
        except Exception as e:
            self.logger.error(f"Error getting status of database environment {name}: {str(e)}")
            return EnvironmentStatus(
                name=name,
                status='unknown',
                error=str(e)
            )


class NativeProcessEnvironmentProvisioner(EnvironmentProvisioner):
    """Provisioner for native process test environments."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the native process environment provisioner.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.base_dir = None
        self.processes = {}
    
    def initialize(self) -> None:
        """Initialize the native process environment provisioner.
        
        Raises:
            ConfigurationError: If the provisioner cannot be initialized.
        """
        self.base_dir = self.config_manager.get('test_env_provisioning.native_process.base_dir', 'environments/native_process')
        
        # Create base directory
        ensure_directory(self.base_dir)
        
        self.logger.info(f"Native process environment provisioner initialized with base directory: {self.base_dir}")
    
    def provision(self, name: str, datasets: List[str], dataset_data: Dict[str, Dict[str, pd.DataFrame]]) -> EnvironmentStatus:
        """Provision a native process test environment.
        
        Args:
            name: Name of the environment.
            datasets: List of dataset names to include.
            dataset_data: Dictionary mapping dataset names to dictionaries mapping table names to DataFrames.
        
        Returns:
            Environment status.
        """
        self.logger.info(f"Provisioning native process environment: {name}")
        
        try:
            # Create environment directory
            env_dir = os.path.join(self.base_dir, name)
            ensure_directory(env_dir)
            
            # Create data directory
            data_dir = os.path.join(env_dir, 'data')
            ensure_directory(data_dir)
            
            # Save datasets as CSV files
            dataset_files = {}
            
            for dataset_name in datasets:
                if dataset_name not in dataset_data:
                    self.logger.warning(f"Dataset {dataset_name} not found, skipping")
                    continue
                
                # Create dataset directory
                dataset_dir = os.path.join(data_dir, dataset_name)
                ensure_directory(dataset_dir)
                
                # Save tables
                table_files = {}
                
                for table_name, df in dataset_data[dataset_name].items():
                    file_path = os.path.join(dataset_dir, f"{table_name}.csv")
                    df.to_csv(file_path, index=False)
                    table_files[table_name] = file_path
                
                dataset_files[dataset_name] = table_files
            
            # Create SQLite database
            db_path = os.path.join(env_dir, f"{name}.db")
            
            # Combine all datasets into a single dictionary
            all_tables = {}
            for dataset_name in datasets:
                if dataset_name not in dataset_data:
                    continue
                
                # Add dataset prefix to table names to avoid conflicts
                for table_name, df in dataset_data[dataset_name].items():
                    all_tables[f"{dataset_name}_{table_name}"] = df
            
            # Create database
            conn = sqlite3.connect(db_path)
            
            for table_name, df in all_tables.items():
                df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            conn.close()
            
            # Create a simple Python HTTP server script
            server_script = os.path.join(env_dir, 'server.py')
            with open(server_script, 'w') as f:
                f.write("""
import http.server
import socketserver
import sqlite3
import json
import os
import sys
from urllib.parse import urlparse, parse_qs

# Get port from command line or use default
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

# Get database path from environment
DB_PATH = os.environ.get('DB_PATH', 'environment.db')

class TestEnvironmentHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # Serve static files
        if parsed_path.path.startswith('/data/'):
            return super().do_GET()
        
        # API endpoints
        if parsed_path.path == '/api/tables':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get list of tables from database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            self.wfile.write(json.dumps({'tables': tables}).encode())
            return
        
        if parsed_path.path.startswith('/api/table/'):
            table_name = parsed_path.path.replace('/api/table/', '')
            
            # Get query parameters
            params = parse_qs(parsed_path.query)
            limit = int(params.get('limit', ['100'])[0])
            offset = int(params.get('offset', ['0'])[0])
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get table data from database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            try:
                cursor.execute(f"SELECT * FROM '{table_name}' LIMIT ? OFFSET ?", (limit, offset))
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                result = {
                    'columns': columns,
                    'rows': rows,
                    'total': cursor.execute(f"SELECT COUNT(*) FROM '{table_name}'").fetchone()[0]
                }
                
                self.wfile.write(json.dumps(result).encode())
            except sqlite3.Error as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            
            conn.close()
            return
        
        # Default: serve index.html
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        index_html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Environment</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .table-selector { margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <h1>Test Environment</h1>
            <div class="table-selector">
                <label for="table-select">Select Table:</label>
                <select id="table-select" onchange="loadTable()"></select>
            </div>
            <div id="table-container">
                <table id="data-table">
                    <thead>
                        <tr id="table-header"></tr>
                    </thead>
                    <tbody id="table-body"></tbody>
                </table>
            </div>
            
            <script>
                // Load tables on page load
                window.onload = function() {
                    fetch('/api/tables')
                        .then(response => response.json())
                        .then(data => {
                            const select = document.getElementById('table-select');
                            data.tables.forEach(table => {
                                const option = document.createElement('option');
                                option.value = table;
                                option.textContent = table;
                                select.appendChild(option);
                            });
                            
                            if (data.tables.length > 0) {
                                loadTable();
                            }
                        });
                };
                
                // Load selected table data
                function loadTable() {
                    const tableName = document.getElementById('table-select').value;
                    
                    fetch(`/api/table/${tableName}?limit=100&offset=0`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.error) {
                                console.error(data.error);
                                return;
                            }
                            
                            // Set header
                            const header = document.getElementById('table-header');
                            header.innerHTML = '';
                            data.columns.forEach(column => {
                                const th = document.createElement('th');
                                th.textContent = column;
                                header.appendChild(th);
                            });
                            
                            // Set body
                            const body = document.getElementById('table-body');
                            body.innerHTML = '';
                            data.rows.forEach(row => {
                                const tr = document.createElement('tr');
                                row.forEach(cell => {
                                    const td = document.createElement('td');
                                    td.textContent = cell;
                                    tr.appendChild(td);
                                });
                                body.appendChild(tr);
                            });
                        });
                }
            </script>
        </body>
        </html>
        '''
        
        self.wfile.write(index_html.encode())

# Set up the server
Handler = TestEnvironmentHandler
httpd = socketserver.TCPServer(("", PORT), Handler)

print(f"Serving at port {PORT}")
print(f"Using database: {DB_PATH}")
print(f"Access the environment at http://localhost:{PORT}")

# Serve until process is killed
httpd.serve_forever()
                """)
            
            # Create a start script
            start_script = os.path.join(env_dir, 'start.sh')
            with open(start_script, 'w') as f:
                f.write(f"""#!/bin/bash
export DB_PATH="{db_path}"
cd "{env_dir}"
python3 server.py 8000
                """)
            
            # Make the start script executable
            os.chmod(start_script, 0o755)
            
            # Start the server process
            process = subprocess.Popen(
                ['bash', start_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=env_dir
            )
            
            # Store the process
            self.processes[name] = process
            
            # Wait for the server to start
            time.sleep(2)
            
            # Check if the process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                self.logger.error(f"Server process failed to start: {stderr.decode()}")
                return EnvironmentStatus(
                    name=name,
                    status='failed',
                    error=f"Server process failed to start: {stderr.decode()}"
                )
            
            # Create metadata file
            metadata = {
                'name': name,
                'type': 'native_process',
                'datasets': datasets,
                'created_at': datetime.datetime.now().isoformat(),
                'status': 'running',
                'db_path': db_path,
                'server_script': server_script,
                'start_script': start_script,
                'port': 8000,
                'pid': process.pid
            }
            
            metadata_path = os.path.join(env_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Return environment status
            return EnvironmentStatus(
                name=name,
                status='running',
                connection_info={
                    'type': 'native_process',
                    'url': 'http://localhost:8000',
                    'port': 8000,
                    'db_path': db_path,
                    'pid': process.pid
                }
            )
        except Exception as e:
            self.logger.error(f"Error provisioning native process environment {name}: {str(e)}")
            return EnvironmentStatus(
                name=name,
                status='failed',
                error=str(e)
            )
    
    def deprovision(self, name: str) -> EnvironmentStatus:
        """Deprovision a native process test environment.
        
        Args:
            name: Name of the environment.
        
        Returns:
            Environment status.
        """
        self.logger.info(f"Deprovisioning native process environment: {name}")
        
        try:
            # Stop the process
            if name in self.processes:
                process = self.processes[name]
                
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    process.wait(timeout=5)
                
                del self.processes[name]
            
            # Get environment directory
            env_dir = os.path.join(self.base_dir, name)
            
            if not os.path.exists(env_dir):
                self.logger.warning(f"Environment directory not found: {env_dir}")
                return EnvironmentStatus(
                    name=name,
                    status='stopped',
                    error=f"Environment directory not found: {env_dir}"
                )
            
            # Get metadata file
            metadata_path = os.path.join(env_dir, 'metadata.json')
            
            if os.path.exists(metadata_path):
                # Read metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Kill the process if it's still running
                pid = metadata.get('pid')
                if pid:
                    try:
                        os.kill(pid, 0)  # Check if process exists
                        os.kill(pid, 15)  # SIGTERM
                    except OSError:
                        pass  # Process doesn't exist
            
            # Remove environment directory
            shutil.rmtree(env_dir)
            
            # Return environment status
            return EnvironmentStatus(
                name=name,
                status='stopped'
            )
        except Exception as e:
            self.logger.error(f"Error deprovisioning native process environment {name}: {str(e)}")
            return EnvironmentStatus(
                name=name,
                status='failed',
                error=str(e)
            )
    
    def get_status(self, name: str) -> EnvironmentStatus:
        """Get the status of a native process test environment.
        
        Args:
            name: Name of the environment.
        
        Returns:
            Environment status.
        """
        self.logger.info(f"Getting status of native process environment: {name}")
        
        try:
            # Check if process is running
            if name in self.processes:
                process = self.processes[name]
                
                if process.poll() is None:  # Process is still running
                    # Get environment directory
                    env_dir = os.path.join(self.base_dir, name)
                    
                    if not os.path.exists(env_dir):
                        self.logger.warning(f"Environment directory not found: {env_dir}")
                        return EnvironmentStatus(
                            name=name,
                            status='unknown',
                            error=f"Environment directory not found: {env_dir}"
                        )
                    
                    # Get metadata file
                    metadata_path = os.path.join(env_dir, 'metadata.json')
                    
                    if not os.path.exists(metadata_path):
                        self.logger.warning(f"Metadata file not found: {metadata_path}")
                        return EnvironmentStatus(
                            name=name,
                            status='unknown',
                            error=f"Metadata file not found: {metadata_path}"
                        )
                    
                    # Read metadata
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Return environment status
                    return EnvironmentStatus(
                        name=name,
                        status='running',
                        connection_info={
                            'type': 'native_process',
                            'url': f"http://localhost:{metadata.get('port', 8000)}",
                            'port': metadata.get('port', 8000),
                            'db_path': metadata.get('db_path'),
                            'pid': process.pid
                        }
                    )
                else:
                    # Process has terminated
                    stdout, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        self.logger.warning(f"Process terminated with error: {stderr.decode()}")
                        return EnvironmentStatus(
                            name=name,
                            status='failed',
                            error=f"Process terminated with error: {stderr.decode()}"
                        )
                    else:
                        return EnvironmentStatus(
                            name=name,
                            status='stopped'
                        )
            
            # Process not found in memory, check if environment directory exists
            env_dir = os.path.join(self.base_dir, name)
            
            if not os.path.exists(env_dir):
                self.logger.warning(f"Environment directory not found: {env_dir}")
                return EnvironmentStatus(
                    name=name,
                    status='stopped',
                    error=f"Environment directory not found: {env_dir}"
                )
            
            # Get metadata file
            metadata_path = os.path.join(env_dir, 'metadata.json')
            
            if not os.path.exists(metadata_path):
                self.logger.warning(f"Metadata file not found: {metadata_path}")
                return EnvironmentStatus(
                    name=name,
                    status='unknown',
                    error=f"Metadata file not found: {metadata_path}"
                )
            
            # Read metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if process is still running
            pid = metadata.get('pid')
            if pid:
                try:
                    os.kill(pid, 0)  # Check if process exists
                    
                    # Process is running but not in our list, add it
                    self.logger.info(f"Found running process for environment {name}, adding to process list")
                    
                    # Return environment status
                    return EnvironmentStatus(
                        name=name,
                        status='running',
                        connection_info={
                            'type': 'native_process',
                            'url': f"http://localhost:{metadata.get('port', 8000)}",
                            'port': metadata.get('port', 8000),
                            'db_path': metadata.get('db_path'),
                            'pid': pid
                        }
                    )
                except OSError:
                    # Process doesn't exist
                    return EnvironmentStatus(
                        name=name,
                        status='stopped',
                        error=f"Process not running"
                    )
            
            # No process information
            return EnvironmentStatus(
                name=name,
                status='unknown',
                error=f"No process information found"
            )
        except Exception as e:
            self.logger.error(f"Error getting status of native process environment {name}: {str(e)}")
            return EnvironmentStatus(
                name=name,
                status='unknown',
                error=str(e)
            )


class TestEnvironmentProvisioningPipeline(Pipeline):
    """Pipeline for test environment provisioning."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the test environment provisioning pipeline.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.file_provisioner = None
        self.database_provisioner = None
        self.native_process_provisioner = None
    
    def initialize(self) -> None:
        """Initialize the test environment provisioning pipeline.
        
        Raises:
            ConfigurationError: If the pipeline cannot be initialized.
        """
        super().initialize()
        
        # Initialize provisioners
        self.file_provisioner = FileEnvironmentProvisioner(self.config_manager)
        self.file_provisioner.initialize()
        
        self.database_provisioner = DatabaseEnvironmentProvisioner(self.config_manager)
        self.database_provisioner.initialize()
        
        self.native_process_provisioner = NativeProcessEnvironmentProvisioner(self.config_manager)
        self.native_process_provisioner.initialize()
        
        # Initialize steps
        self.steps = [
            PipelineStep(self.prepare_environment_data),
            PipelineStep(self.provision_environments),
            PipelineStep(self.save_results)
        ]
        
        self.logger.info("Test environment provisioning pipeline initialized")
    
    def prepare_environment_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for environment provisioning.
        
        Args:
            data: Input data with datasets.
        
        Returns:
            Updated data with environment configurations.
        
        Raises:
            ProcessingError: If preparation fails.
        """
        self.logger.info("Preparing environment data")
        
        try:
            datasets = data.get('datasets', {})
            
            if not datasets:
                self.logger.warning("No datasets found for environment provisioning")
                return data
            
            # Get environment configurations
            env_configs = self.config_manager.get('test_env_provisioning.environments', [])
            
            if not env_configs:
                self.logger.warning("No environment configurations found")
                return data
            
            # Validate environment configurations
            valid_env_configs = []
            
            for config in env_configs:
                name = config.get('name')
                env_type = config.get('type')
                env_datasets = config.get('datasets', [])
                
                if not name:
                    self.logger.warning("Environment configuration missing name, skipping")
                    continue
                
                if not env_type:
                    self.logger.warning(f"Environment configuration {name} missing type, skipping")
                    continue
                
                if not env_datasets:
                    self.logger.warning(f"Environment configuration {name} has no datasets, skipping")
                    continue
                
                # Check if datasets exist
                missing_datasets = [ds for ds in env_datasets if ds not in datasets]
                
                if missing_datasets:
                    self.logger.warning(f"Environment configuration {name} references missing datasets: {missing_datasets}")
                    continue
                
                valid_env_configs.append(config)
            
            # Update data
            result = data.copy()
            result['environment_configs'] = valid_env_configs
            
            self.logger.info(f"Prepared {len(valid_env_configs)} environment configurations")
            
            return result
        except Exception as e:
            self.logger.error(f"Error preparing environment data: {str(e)}")
            raise ProcessingError(f"Error preparing environment data: {str(e)}")
    
    def provision_environments(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Provision test environments.
        
        Args:
            data: Input data with environment configurations.
        
        Returns:
            Updated data with provisioned environments.
        
        Raises:
            ProcessingError: If provisioning fails.
        """
        self.logger.info("Provisioning test environments")
        
        try:
            env_configs = data.get('environment_configs', [])
            datasets = data.get('datasets', {})
            
            if not env_configs:
                self.logger.warning("No environment configurations found for provisioning")
                return data
            
            # Provision environments
            environments = {}
            
            for config in env_configs:
                name = config.get('name')
                env_type = config.get('type')
                env_datasets = config.get('datasets', [])
                
                self.logger.info(f"Provisioning environment {name} of type {env_type}")
                
                # Get dataset data
                dataset_data = {ds: datasets.get(ds, {}) for ds in env_datasets}
                
                # Provision environment based on type
                if env_type == 'file':
                    status = self.file_provisioner.provision(name, env_datasets, dataset_data)
                elif env_type == 'database':
                    status = self.database_provisioner.provision(name, env_datasets, dataset_data)
                elif env_type == 'native_process':
                    status = self.native_process_provisioner.provision(name, env_datasets, dataset_data)
                else:
                    self.logger.warning(f"Unsupported environment type: {env_type}")
                    status = EnvironmentStatus(
                        name=name,
                        status='failed',
                        error=f"Unsupported environment type: {env_type}"
                    )
                
                environments[name] = status
            
            # Update data
            result = data.copy()
            result['environments'] = environments
            
            # Log summary
            successful = sum(1 for status in environments.values() if status.status == 'running')
            failed = sum(1 for status in environments.values() if status.status == 'failed')
            
            self.logger.info(f"Provisioned {len(environments)} environments: {successful} successful, {failed} failed")
            
            return result
        except Exception as e:
            self.logger.error(f"Error provisioning environments: {str(e)}")
            raise ProcessingError(f"Error provisioning environments: {str(e)}")
    
    def save_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save provisioning results.
        
        Args:
            data: Input data with provisioned environments.
        
        Returns:
            Updated data with saved results.
        
        Raises:
            ProcessingError: If saving results fails.
        """
        self.logger.info("Saving provisioning results")
        
        try:
            environments = data.get('environments', {})
            
            if not environments:
                self.logger.warning("No environments found to save")
                return data
            
            # Create output directory
            results_dir = os.path.join(self.output_dir, 'results')
            ensure_directory(results_dir)
            
            # Save environment statuses
            env_statuses = {}
            
            for name, status in environments.items():
                if hasattr(status, 'to_dict'):
                    env_statuses[name] = status.to_dict()
                else:
                    env_statuses[name] = status
            
            results_file = os.path.join(results_dir, 'environment_statuses.json')
            save_json(env_statuses, results_file)
            
            # Save summary
            summary = {
                'environment_count': len(environments),
                'successful_count': sum(1 for status in environments.values() if status.status == 'running'),
                'failed_count': sum(1 for status in environments.values() if status.status == 'failed'),
                'environments': {name: {'status': status.status, 'type': status.connection_info.get('type') if status.connection_info else None} for name, status in environments.items()}
            }
            
            summary_file = os.path.join(results_dir, 'provisioning_summary.json')
            save_json(summary, summary_file)
            
            # Create connection information file
            connections = {}
            
            for name, status in environments.items():
                if status.status == 'running' and status.connection_info:
                    connections[name] = status.connection_info
            
            connections_file = os.path.join(results_dir, 'environment_connections.json')
            save_json(connections, connections_file)
            
            # Update data
            result = data.copy()
            result['provisioning_results'] = {
                'results_file': results_file,
                'summary_file': summary_file,
                'connections_file': connections_file
            }
            
            self.logger.info("Provisioning results saved")
            return result
        except Exception as e:
            self.logger.error(f"Error saving provisioning results: {str(e)}")
            raise ProcessingError(f"Error saving provisioning results: {str(e)}")
    
    def deprovision_environment(self, name: str) -> EnvironmentStatus:
        """Deprovision a test environment.
        
        Args:
            name: Name of the environment.
        
        Returns:
            Environment status.
        """
        self.logger.info(f"Deprovisioning environment: {name}")
        
        try:
            # Get environment status
            file_status = self.file_provisioner.get_status(name)
            
            if file_status.status == 'running':
                return self.file_provisioner.deprovision(name)
            
            database_status = self.database_provisioner.get_status(name)
            
            if database_status.status == 'running':
                return self.database_provisioner.deprovision(name)
            
            native_process_status = self.native_process_provisioner.get_status(name)
            
            if native_process_status.status == 'running':
                return self.native_process_provisioner.deprovision(name)
            
            # Environment not found or not running
            self.logger.warning(f"Environment {name} not found or not running")
            return EnvironmentStatus(
                name=name,
                status='unknown',
                error=f"Environment not found or not running"
            )
        except Exception as e:
            self.logger.error(f"Error deprovisioning environment {name}: {str(e)}")
            return EnvironmentStatus(
                name=name,
                status='failed',
                error=str(e)
            )
    
    def get_environment_status(self, name: str) -> EnvironmentStatus:
        """Get the status of a test environment.
        
        Args:
            name: Name of the environment.
        
        Returns:
            Environment status.
        """
        self.logger.info(f"Getting status of environment: {name}")
        
        try:
            # Check each provisioner
            file_status = self.file_provisioner.get_status(name)
            
            if file_status.status != 'unknown' and file_status.status != 'stopped':
                return file_status
            
            database_status = self.database_provisioner.get_status(name)
            
            if database_status.status != 'unknown' and database_status.status != 'stopped':
                return database_status
            
            native_process_status = self.native_process_provisioner.get_status(name)
            
            if native_process_status.status != 'unknown' and native_process_status.status != 'stopped':
                return native_process_status
            
            # Environment not found or stopped
            if file_status.status == 'stopped':
                return file_status
            
            if database_status.status == 'stopped':
                return database_status
            
            if native_process_status.status == 'stopped':
                return native_process_status
            
            # Environment not found
            self.logger.warning(f"Environment {name} not found")
            return EnvironmentStatus(
                name=name,
                status='unknown',
                error=f"Environment not found"
            )
        except Exception as e:
            self.logger.error(f"Error getting status of environment {name}: {str(e)}")
            return EnvironmentStatus(
                name=name,
                status='unknown',
                error=str(e)
            )

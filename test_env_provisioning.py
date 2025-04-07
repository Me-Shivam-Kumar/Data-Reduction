"""
Test Environment Provisioning Module for the Data Warehouse Subsampling Framework.

This module implements the fifth layer of the data subsampling architecture,
responsible for using virtualization to minimize storage requirements,
implementing copy-on-write for efficient environment creation,
and providing on-demand test environment provisioning.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import shutil
import subprocess
import tempfile
import uuid
import docker
import yaml
import time

from ..common.base import Component, ConfigManager, PipelineStep, Pipeline, ProcessingError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for a test environment."""
    name: str
    type: str  # 'docker', 'file', 'database'
    datasets: List[str]
    parameters: Dict[str, Any]
    created_at: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the environment configuration to a dictionary.
        
        Returns:
            Dictionary representation of the environment configuration.
        """
        return {
            'name': self.name,
            'type': self.type,
            'datasets': self.datasets,
            'parameters': self.parameters,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentConfig':
        """Create an EnvironmentConfig from a dictionary.
        
        Args:
            data: Dictionary with environment configuration information.
        
        Returns:
            EnvironmentConfig instance.
        """
        created_at = None
        if data.get('created_at'):
            try:
                created_at = datetime.fromisoformat(data['created_at'])
            except (ValueError, TypeError):
                created_at = None
        
        return cls(
            name=data.get('name', ''),
            type=data.get('type', ''),
            datasets=data.get('datasets', []),
            parameters=data.get('parameters', {}),
            created_at=created_at
        )


@dataclass
class EnvironmentStatus:
    """Status of a test environment."""
    name: str
    status: str  # 'creating', 'running', 'stopped', 'failed'
    start_time: datetime = None
    end_time: datetime = None
    error_message: str = None
    connection_info: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the environment status to a dictionary.
        
        Returns:
            Dictionary representation of the environment status.
        """
        return {
            'name': self.name,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error_message': self.error_message,
            'connection_info': self.connection_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentStatus':
        """Create an EnvironmentStatus from a dictionary.
        
        Args:
            data: Dictionary with environment status information.
        
        Returns:
            EnvironmentStatus instance.
        """
        start_time = None
        if data.get('start_time'):
            try:
                start_time = datetime.fromisoformat(data['start_time'])
            except (ValueError, TypeError):
                start_time = None
        
        end_time = None
        if data.get('end_time'):
            try:
                end_time = datetime.fromisoformat(data['end_time'])
            except (ValueError, TypeError):
                end_time = None
        
        return cls(
            name=data.get('name', ''),
            status=data.get('status', ''),
            start_time=start_time,
            end_time=end_time,
            error_message=data.get('error_message'),
            connection_info=data.get('connection_info', {})
        )


class EnvironmentManager(Component):
    """Component for managing test environments."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the environment manager.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.output_dir = None
        self.environments_dir = None
        self.environments = {}
    
    def initialize(self) -> None:
        """Initialize the environment manager.
        
        Raises:
            ConfigurationError: If the manager cannot be initialized.
        """
        # Create output directory
        self.output_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'test_env_provisioning'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create environments directory
        self.environments_dir = os.path.join(self.output_dir, 'environments')
        os.makedirs(self.environments_dir, exist_ok=True)
        
        # Load existing environments
        self._load_environments()
        
        self.logger.info(f"Environment manager initialized with {len(self.environments)} environments")
    
    def validate(self) -> bool:
        """Validate the environment manager configuration and state.
        
        Returns:
            True if the manager is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if not os.path.exists(self.output_dir):
            raise ValidationError(f"Output directory does not exist: {self.output_dir}")
        
        if not os.path.exists(self.environments_dir):
            raise ValidationError(f"Environments directory does not exist: {self.environments_dir}")
        
        return True
    
    def create_environment(self, config: EnvironmentConfig) -> EnvironmentStatus:
        """Create a new test environment.
        
        Args:
            config: Environment configuration.
        
        Returns:
            Status of the created environment.
        
        Raises:
            ProcessingError: If environment creation fails.
        """
        self.logger.info(f"Creating environment '{config.name}' of type '{config.type}'")
        
        # Check if environment already exists
        if config.name in self.environments:
            raise ProcessingError(f"Environment '{config.name}' already exists")
        
        # Create environment directory
        env_dir = os.path.join(self.environments_dir, config.name)
        os.makedirs(env_dir, exist_ok=True)
        
        # Save configuration
        config_file = os.path.join(env_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Create initial status
        status = EnvironmentStatus(
            name=config.name,
            status='creating',
            start_time=datetime.now()
        )
        
        # Save status
        status_file = os.path.join(env_dir, 'status.json')
        with open(status_file, 'w') as f:
            json.dump(status.to_dict(), f, indent=2)
        
        # Add to environments
        self.environments[config.name] = (config, status)
        
        # Create environment based on type
        try:
            if config.type == 'docker':
                status = self._create_docker_environment(config, env_dir)
            elif config.type == 'file':
                status = self._create_file_environment(config, env_dir)
            elif config.type == 'database':
                status = self._create_database_environment(config, env_dir)
            else:
                raise ProcessingError(f"Unsupported environment type: {config.type}")
            
            # Update status
            status.status = 'running'
            status.end_time = datetime.now()
            
            # Save status
            with open(status_file, 'w') as f:
                json.dump(status.to_dict(), f, indent=2)
            
            # Update environments
            self.environments[config.name] = (config, status)
            
            self.logger.info(f"Environment '{config.name}' created successfully")
            return status
        except Exception as e:
            # Update status
            status.status = 'failed'
            status.end_time = datetime.now()
            status.error_message = str(e)
            
            # Save status
            with open(status_file, 'w') as f:
                json.dump(status.to_dict(), f, indent=2)
            
            # Update environments
            self.environments[config.name] = (config, status)
            
            self.logger.error(f"Error creating environment '{config.name}': {str(e)}")
            raise ProcessingError(f"Error creating environment '{config.name}': {str(e)}")
    
    def get_environment_status(self, name: str) -> EnvironmentStatus:
        """Get the status of a test environment.
        
        Args:
            name: Name of the environment.
        
        Returns:
            Status of the environment.
        
        Raises:
            ProcessingError: If the environment does not exist.
        """
        if name not in self.environments:
            raise ProcessingError(f"Environment '{name}' does not exist")
        
        return self.environments[name][1]
    
    def stop_environment(self, name: str) -> EnvironmentStatus:
        """Stop a test environment.
        
        Args:
            name: Name of the environment.
        
        Returns:
            Status of the environment.
        
        Raises:
            ProcessingError: If the environment does not exist or cannot be stopped.
        """
        self.logger.info(f"Stopping environment '{name}'")
        
        if name not in self.environments:
            raise ProcessingError(f"Environment '{name}' does not exist")
        
        config, status = self.environments[name]
        
        # Check if environment is already stopped
        if status.status == 'stopped':
            return status
        
        # Stop environment based on type
        try:
            if config.type == 'docker':
                status = self._stop_docker_environment(config, status)
            elif config.type == 'file':
                status = self._stop_file_environment(config, status)
            elif config.type == 'database':
                status = self._stop_database_environment(config, status)
            else:
                raise ProcessingError(f"Unsupported environment type: {config.type}")
            
            # Update status
            status.status = 'stopped'
            status.end_time = datetime.now()
            
            # Save status
            env_dir = os.path.join(self.environments_dir, name)
            status_file = os.path.join(env_dir, 'status.json')
            with open(status_file, 'w') as f:
                json.dump(status.to_dict(), f, indent=2)
            
            # Update environments
            self.environments[name] = (config, status)
            
            self.logger.info(f"Environment '{name}' stopped successfully")
            return status
        except Exception as e:
            # Update status
            status.status = 'failed'
            status.end_time = datetime.now()
            status.error_message = str(e)
            
            # Save status
            env_dir = os.path.join(self.environments_dir, name)
            status_file = os.path.join(env_dir, 'status.json')
            with open(status_file, 'w') as f:
                json.dump(status.to_dict(), f, indent=2)
            
            # Update environments
            self.environments[name] = (config, status)
            
            self.logger.error(f"Error stopping environment '{name}': {str(e)}")
            raise ProcessingError(f"Error stopping environment '{name}': {str(e)}")
    
    def delete_environment(self, name: str) -> None:
        """Delete a test environment.
        
        Args:
            name: Name of the environment.
        
        Raises:
            ProcessingError: If the environment does not exist or cannot be deleted.
        """
        self.logger.info(f"Deleting environment '{name}'")
        
        if name not in self.environments:
            raise ProcessingError(f"Environment '{name}' does not exist")
        
        config, status = self.environments[name]
        
        # Stop environment if running
        if status.status == 'running':
            try:
                self.stop_environment(name)
            except Exception as e:
                self.logger.error(f"Error stopping environment '{name}' before deletion: {str(e)}")
        
        # Delete environment based on type
        try:
            if config.type == 'docker':
                self._delete_docker_environment(config)
            elif config.type == 'file':
                self._delete_file_environment(config)
            elif config.type == 'database':
                self._delete_database_environment(config)
            else:
                raise ProcessingError(f"Unsupported environment type: {config.type}")
            
            # Delete environment directory
            env_dir = os.path.join(self.environments_dir, name)
            shutil.rmtree(env_dir, ignore_errors=True)
            
            # Remove from environments
            del self.environments[name]
            
            self.logger.info(f"Environment '{name}' deleted successfully")
        except Exception as e:
            self.logger.error(f"Error deleting environment '{name}': {str(e)}")
            raise ProcessingError(f"Error deleting environment '{name}': {str(e)}")
    
    def list_environments(self) -> List[Dict[str, Any]]:
        """List all test environments.
        
        Returns:
            List of environment information dictionaries.
        """
        result = []
        
        for name, (config, status) in self.environments.items():
            result.append({
                'name': name,
                'type': config.type,
                'status': status.status,
                'datasets': config.datasets,
                'start_time': status.start_time.isoformat() if status.start_time else None,
                'end_time': status.end_time.isoformat() if status.end_time else None,
                'connection_info': status.connection_info
            })
        
        return result
    
    def _load_environments(self) -> None:
        """Load existing environments from disk."""
        self.environments = {}
        
        # Get environment directories
        for env_name in os.listdir(self.environments_dir):
            env_dir = os.path.join(self.environments_dir, env_name)
            
            if not os.path.isdir(env_dir):
                continue
            
            # Load configuration
            config_file = os.path.join(env_dir, 'config.json')
            if not os.path.exists(config_file):
                continue
            
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    config = EnvironmentConfig.from_dict(config_data)
            except Exception as e:
                self.logger.error(f"Error loading configuration for environment '{env_name}': {str(e)}")
                continue
            
            # Load status
            status_file = os.path.join(env_dir, 'status.json')
            if not os.path.exists(status_file):
                continue
            
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                    status = EnvironmentStatus.from_dict(status_data)
            except Exception as e:
                self.logger.error(f"Error loading status for environment '{env_name}': {str(e)}")
                continue
            
            # Add to environments
            self.environments[env_name] = (config, status)
    
    def _create_docker_environment(self, config: EnvironmentConfig, env_dir: str) -> EnvironmentStatus:
        """Create a Docker-based test environment.
        
        Args:
            config: Environment configuration.
            env_dir: Directory for environment files.
        
        Returns:
            Status of the created environment.
        
        Raises:
            ProcessingError: If environment creation fails.
        """
        self.logger.info(f"Creating Docker environment '{config.name}'")
        
        # Get Docker parameters
        image = config.parameters.get('image', 'postgres:latest')
        ports = config.parameters.get('ports', {})
        environment = config.parameters.get('environment', {})
        volumes = config.parameters.get('volumes', [])
        
        # Create Docker client
        try:
            client = docker.from_env()
        except Exception as e:
            raise ProcessingError(f"Error creating Docker client: {str(e)}")
        
        # Create container
        try:
            container = client.containers.run(
                image=image,
                name=f"dwsf_{config.name}",
                ports=ports,
                environment=environment,
                volumes=volumes,
                detach=True
            )
            
            # Wait for container to start
            time.sleep(5)
            
            # Get container info
            container_info = client.containers.get(container.id).attrs
            
            # Create connection info
            connection_info = {
                'container_id': container.id,
                'container_name': container_info['Name'],
                'ports': {}
            }
            
            # Get port mappings
            for container_port, host_ports in container_info['NetworkSettings']['Ports'].items():
                if host_ports:
                    connection_info['ports'][container_port] = host_ports[0]['HostPort']
            
            # Create status
            status = EnvironmentStatus(
                name=config.name,
                status='running',
                start_time=datetime.now(),
                connection_info=connection_info
            )
            
            # Load datasets
            self._load_datasets_to_docker(config, container, connection_info)
            
            return status
        except Exception as e:
            raise ProcessingError(f"Error creating Docker container: {str(e)}")
    
    def _stop_docker_environment(self, config: EnvironmentConfig, status: EnvironmentStatus) -> EnvironmentStatus:
        """Stop a Docker-based test environment.
        
        Args:
            config: Environment configuration.
            status: Current environment status.
        
        Returns:
            Updated environment status.
        
        Raises:
            ProcessingError: If environment stopping fails.
        """
        self.logger.info(f"Stopping Docker environment '{config.name}'")
        
        # Get container ID
        container_id = status.connection_info.get('container_id')
        
        if not container_id:
            raise ProcessingError(f"Container ID not found for environment '{config.name}'")
        
        # Create Docker client
        try:
            client = docker.from_env()
        except Exception as e:
            raise ProcessingError(f"Error creating Docker client: {str(e)}")
        
        # Stop container
        try:
            container = client.containers.get(container_id)
            container.stop()
            
            # Update status
            status.status = 'stopped'
            status.end_time = datetime.now()
            
            return status
        except Exception as e:
            raise ProcessingError(f"Error stopping Docker container: {str(e)}")
    
    def _delete_docker_environment(self, config: EnvironmentConfig) -> None:
        """Delete a Docker-based test environment.
        
        Args:
            config: Environment configuration.
        
        Raises:
            ProcessingError: If environment deletion fails.
        """
        self.logger.info(f"Deleting Docker environment '{config.name}'")
        
        # Create Docker client
        try:
            client = docker.from_env()
        except Exception as e:
            raise ProcessingError(f"Error creating Docker client: {str(e)}")
        
        # Delete container
        try:
            container = client.containers.get(f"dwsf_{config.name}")
            container.remove(force=True)
        except docker.errors.NotFound:
            # Container already deleted
            pass
        except Exception as e:
            raise ProcessingError(f"Error deleting Docker container: {str(e)}")
    
    def _load_datasets_to_docker(self, config: EnvironmentConfig, container: Any, connection_info: Dict[str, Any]) -> None:
        """Load datasets to a Docker container.
        
        Args:
            config: Environment configuration.
            container: Docker container.
            connection_info: Connection information.
        
        Raises:
            ProcessingError: If dataset loading fails.
        """
        self.logger.info(f"Loading datasets to Docker environment '{config.name}'")
        
        # Get datasets
        datasets = config.datasets
        
        if not datasets:
            self.logger.info(f"No datasets to load for environment '{config.name}'")
            return
        
        # Get export paths
        export_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'data_integration',
            'exports'
        )
        
        # Get database parameters
        db_type = config.parameters.get('db_type', 'postgresql')
        db_name = config.parameters.get('db_name', 'testdb')
        db_user = config.parameters.get('db_user', 'postgres')
        db_password = config.parameters.get('db_password', 'postgres')
        
        # Load datasets based on database type
        for dataset in datasets:
            dataset_dir = os.path.join(export_dir, dataset)
            
            if not os.path.exists(dataset_dir):
                self.logger.warning(f"Dataset directory not found: {dataset_dir}")
                continue
            
            if db_type == 'postgresql':
                self._load_dataset_to_postgresql(dataset_dir, container, connection_info, db_name, db_user, db_password)
            elif db_type == 'mysql':
                self._load_dataset_to_mysql(dataset_dir, container, connection_info, db_name, db_user, db_password)
            else:
                self.logger.warning(f"Unsupported database type: {db_type}")
    
    def _load_dataset_to_postgresql(self, dataset_dir: str, container: Any, connection_info: Dict[str, Any],
                                  db_name: str, db_user: str, db_password: str) -> None:
        """Load a dataset to a PostgreSQL database in a Docker container.
        
        Args:
            dataset_dir: Directory containing dataset files.
            container: Docker container.
            connection_info: Connection information.
            db_name: Database name.
            db_user: Database user.
            db_password: Database password.
        
        Raises:
            ProcessingError: If dataset loading fails.
        """
        self.logger.info(f"Loading dataset from {dataset_dir} to PostgreSQL database")
        
        # Get CSV files
        csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
        
        if not csv_files:
            self.logger.warning(f"No CSV files found in dataset directory: {dataset_dir}")
            return
        
        # Create temporary directory for SQL files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create SQL files for each table
            for csv_file in csv_files:
                table_name = os.path.splitext(csv_file)[0]
                csv_path = os.path.join(dataset_dir, csv_file)
                
                # Read CSV file
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    self.logger.error(f"Error reading CSV file {csv_path}: {str(e)}")
                    continue
                
                # Create SQL file
                sql_path = os.path.join(temp_dir, f"{table_name}.sql")
                
                with open(sql_path, 'w') as f:
                    # Create table
                    f.write(f"DROP TABLE IF EXISTS {table_name};\n")
                    f.write(f"CREATE TABLE {table_name} (\n")
                    
                    # Add columns
                    columns = []
                    for col in df.columns:
                        # Determine column type
                        if pd.api.types.is_numeric_dtype(df[col]):
                            if pd.api.types.is_integer_dtype(df[col]):
                                col_type = "INTEGER"
                            else:
                                col_type = "NUMERIC"
                        elif pd.api.types.is_datetime64_dtype(df[col]):
                            col_type = "TIMESTAMP"
                        elif pd.api.types.is_bool_dtype(df[col]):
                            col_type = "BOOLEAN"
                        else:
                            col_type = "TEXT"
                        
                        columns.append(f"    \"{col}\" {col_type}")
                    
                    f.write(",\n".join(columns))
                    f.write("\n);\n\n")
                    
                    # Insert data
                    for _, row in df.iterrows():
                        values = []
                        for val in row:
                            if pd.isna(val):
                                values.append("NULL")
                            elif isinstance(val, (int, float)):
                                values.append(str(val))
                            elif isinstance(val, bool):
                                values.append("TRUE" if val else "FALSE")
                            else:
                                values.append(f"""'{str(val).replace("'", "''")}'""")

                        
                        f.write(f"INSERT INTO {table_name} VALUES ({', '.join(values)});\n")
                
                # Copy SQL file to container
                container.exec_run(f"mkdir -p /tmp/dwsf")
                with open(sql_path, 'rb') as f:
                    container.put_archive("/tmp/dwsf", f.read())
                
                # Execute SQL file
                container.exec_run(
                    f"psql -U {db_user} -d {db_name} -f /tmp/dwsf/{table_name}.sql",
                    environment={"PGPASSWORD": db_password}
                )
    
    def _load_dataset_to_mysql(self, dataset_dir: str, container: Any, connection_info: Dict[str, Any],
                             db_name: str, db_user: str, db_password: str) -> None:
        """Load a dataset to a MySQL database in a Docker container.
        
        Args:
            dataset_dir: Directory containing dataset files.
            container: Docker container.
            connection_info: Connection information.
            db_name: Database name.
            db_user: Database user.
            db_password: Database password.
        
        Raises:
            ProcessingError: If dataset loading fails.
        """
        self.logger.info(f"Loading dataset from {dataset_dir} to MySQL database")
        
        # Get CSV files
        csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
        
        if not csv_files:
            self.logger.warning(f"No CSV files found in dataset directory: {dataset_dir}")
            return
        
        # Create temporary directory for SQL files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create SQL files for each table
            for csv_file in csv_files:
                table_name = os.path.splitext(csv_file)[0]
                csv_path = os.path.join(dataset_dir, csv_file)
                
                # Read CSV file
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    self.logger.error(f"Error reading CSV file {csv_path}: {str(e)}")
                    continue
                
                # Create SQL file
                sql_path = os.path.join(temp_dir, f"{table_name}.sql")
                
                with open(sql_path, 'w') as f:
                    # Create table
                    f.write(f"DROP TABLE IF EXISTS {table_name};\n")
                    f.write(f"CREATE TABLE {table_name} (\n")
                    
                    # Add columns
                    columns = []
                    for col in df.columns:
                        # Determine column type
                        if pd.api.types.is_numeric_dtype(df[col]):
                            if pd.api.types.is_integer_dtype(df[col]):
                                col_type = "INT"
                            else:
                                col_type = "DOUBLE"
                        elif pd.api.types.is_datetime64_dtype(df[col]):
                            col_type = "DATETIME"
                        elif pd.api.types.is_bool_dtype(df[col]):
                            col_type = "BOOLEAN"
                        else:
                            col_type = "TEXT"
                        
                        columns.append(f"    `{col}` {col_type}")
                    
                    f.write(",\n".join(columns))
                    f.write("\n);\n\n")
                    
                    # Insert data
                    for _, row in df.iterrows():
                        values = []
                        for val in row:
                            if pd.isna(val):
                                values.append("NULL")
                            elif isinstance(val, (int, float)):
                                values.append(str(val))
                            elif isinstance(val, bool):
                                values.append("1" if val else "0")
                            else:
                               values.append(f"""'{str(val).replace("'", "''")}'""")

                        
                        f.write(f"INSERT INTO {table_name} VALUES ({', '.join(values)});\n")
                
                # Copy SQL file to container
                container.exec_run(f"mkdir -p /tmp/dwsf")
                with open(sql_path, 'rb') as f:
                    container.put_archive("/tmp/dwsf", f.read())
                
                # Execute SQL file
                container.exec_run(
                    f"mysql -u {db_user} -p{db_password} {db_name} < /tmp/dwsf/{table_name}.sql"
                )
    
    def _create_file_environment(self, config: EnvironmentConfig, env_dir: str) -> EnvironmentStatus:
        """Create a file-based test environment.
        
        Args:
            config: Environment configuration.
            env_dir: Directory for environment files.
        
        Returns:
            Status of the created environment.
        
        Raises:
            ProcessingError: If environment creation fails.
        """
        self.logger.info(f"Creating file environment '{config.name}'")
        
        # Get file parameters
        base_dir = config.parameters.get('base_dir', os.path.join(env_dir, 'data'))
        
        # Create data directory
        os.makedirs(base_dir, exist_ok=True)
        
        # Create connection info
        connection_info = {
            'base_dir': base_dir
        }
        
        # Create status
        status = EnvironmentStatus(
            name=config.name,
            status='running',
            start_time=datetime.now(),
            connection_info=connection_info
        )
        
        # Load datasets
        self._load_datasets_to_files(config, base_dir)
        
        return status
    
    def _stop_file_environment(self, config: EnvironmentConfig, status: EnvironmentStatus) -> EnvironmentStatus:
        """Stop a file-based test environment.
        
        Args:
            config: Environment configuration.
            status: Current environment status.
        
        Returns:
            Updated environment status.
        
        Raises:
            ProcessingError: If environment stopping fails.
        """
        self.logger.info(f"Stopping file environment '{config.name}'")
        
        # File environments don't need to be stopped
        status.status = 'stopped'
        status.end_time = datetime.now()
        
        return status
    
    def _delete_file_environment(self, config: EnvironmentConfig) -> None:
        """Delete a file-based test environment.
        
        Args:
            config: Environment configuration.
        
        Raises:
            ProcessingError: If environment deletion fails.
        """
        self.logger.info(f"Deleting file environment '{config.name}'")
        
        # Get base directory
        base_dir = config.parameters.get('base_dir')
        
        if base_dir and os.path.exists(base_dir):
            shutil.rmtree(base_dir, ignore_errors=True)
    
    def _load_datasets_to_files(self, config: EnvironmentConfig, base_dir: str) -> None:
        """Load datasets to a file-based environment.
        
        Args:
            config: Environment configuration.
            base_dir: Base directory for environment files.
        
        Raises:
            ProcessingError: If dataset loading fails.
        """
        self.logger.info(f"Loading datasets to file environment '{config.name}'")
        
        # Get datasets
        datasets = config.datasets
        
        if not datasets:
            self.logger.info(f"No datasets to load for environment '{config.name}'")
            return
        
        # Get export paths
        export_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'data_integration',
            'exports'
        )
        
        # Copy datasets
        for dataset in datasets:
            dataset_dir = os.path.join(export_dir, dataset)
            
            if not os.path.exists(dataset_dir):
                self.logger.warning(f"Dataset directory not found: {dataset_dir}")
                continue
            
            # Create dataset directory
            dataset_output_dir = os.path.join(base_dir, dataset)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # Copy files
            for file_name in os.listdir(dataset_dir):
                src_path = os.path.join(dataset_dir, file_name)
                dst_path = os.path.join(dataset_output_dir, file_name)
                
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
    
    def _create_database_environment(self, config: EnvironmentConfig, env_dir: str) -> EnvironmentStatus:
        """Create a database-based test environment.
        
        Args:
            config: Environment configuration.
            env_dir: Directory for environment files.
        
        Returns:
            Status of the created environment.
        
        Raises:
            ProcessingError: If environment creation fails.
        """
        self.logger.info(f"Creating database environment '{config.name}'")
        
        # Get database parameters
        db_type = config.parameters.get('db_type', 'sqlite')
        db_name = config.parameters.get('db_name', 'testdb')
        
        # Create connection info
        connection_info = {
            'db_type': db_type,
            'db_name': db_name
        }
        
        if db_type == 'sqlite':
            # Create SQLite database
            db_path = os.path.join(env_dir, f"{db_name}.db")
            connection_info['db_path'] = db_path
            
            # Create database
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.close()
            
            # Load datasets
            self._load_datasets_to_sqlite(config, db_path)
        else:
            raise ProcessingError(f"Unsupported database type: {db_type}")
        
        # Create status
        status = EnvironmentStatus(
            name=config.name,
            status='running',
            start_time=datetime.now(),
            connection_info=connection_info
        )
        
        return status
    
    def _stop_database_environment(self, config: EnvironmentConfig, status: EnvironmentStatus) -> EnvironmentStatus:
        """Stop a database-based test environment.
        
        Args:
            config: Environment configuration.
            status: Current environment status.
        
        Returns:
            Updated environment status.
        
        Raises:
            ProcessingError: If environment stopping fails.
        """
        self.logger.info(f"Stopping database environment '{config.name}'")
        
        # Database environments don't need to be stopped
        status.status = 'stopped'
        status.end_time = datetime.now()
        
        return status
    
    def _delete_database_environment(self, config: EnvironmentConfig) -> None:
        """Delete a database-based test environment.
        
        Args:
            config: Environment configuration.
        
        Raises:
            ProcessingError: If environment deletion fails.
        """
        self.logger.info(f"Deleting database environment '{config.name}'")
        
        # Get database parameters
        db_type = config.parameters.get('db_type', 'sqlite')
        
        if db_type == 'sqlite':
            # Get database path
            db_path = config.parameters.get('db_path')
            
            if db_path and os.path.exists(db_path):
                os.remove(db_path)
        else:
            self.logger.warning(f"Unsupported database type: {db_type}")
    
    def _load_datasets_to_sqlite(self, config: EnvironmentConfig, db_path: str) -> None:
        """Load datasets to a SQLite database.
        
        Args:
            config: Environment configuration.
            db_path: Path to the SQLite database file.
        
        Raises:
            ProcessingError: If dataset loading fails.
        """
        self.logger.info(f"Loading datasets to SQLite database '{db_path}'")
        
        # Get datasets
        datasets = config.datasets
        
        if not datasets:
            self.logger.info(f"No datasets to load for environment '{config.name}'")
            return
        
        # Get export paths
        export_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'data_integration',
            'exports'
        )
        
        # Connect to database
        import sqlite3
        conn = sqlite3.connect(db_path)
        
        try:
            # Load datasets
            for dataset in datasets:
                dataset_dir = os.path.join(export_dir, dataset)
                
                if not os.path.exists(dataset_dir):
                    self.logger.warning(f"Dataset directory not found: {dataset_dir}")
                    continue
                
                # Get CSV files
                csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
                
                if not csv_files:
                    self.logger.warning(f"No CSV files found in dataset directory: {dataset_dir}")
                    continue
                
                # Load each table
                for csv_file in csv_files:
                    table_name = os.path.splitext(csv_file)[0]
                    csv_path = os.path.join(dataset_dir, csv_file)
                    
                    # Read CSV file
                    try:
                        df = pd.read_csv(csv_path)
                    except Exception as e:
                        self.logger.error(f"Error reading CSV file {csv_path}: {str(e)}")
                        continue
                    
                    # Write to SQLite
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
        finally:
            conn.close()


class DockerEnvironmentProvisioner(Component):
    """Component for provisioning Docker-based test environments."""
    
    def __init__(self, config_manager: ConfigManager, environment_manager: EnvironmentManager):
        """Initialize the Docker environment provisioner.
        
        Args:
            config_manager: Configuration manager instance.
            environment_manager: EnvironmentManager instance.
        """
        super().__init__(config_manager)
        self.environment_manager = environment_manager
        self.docker_config = None
    
    def initialize(self) -> None:
        """Initialize the Docker environment provisioner.
        
        Raises:
            ConfigurationError: If the provisioner cannot be initialized.
        """
        # Get Docker configuration
        self.docker_config = self.config_manager.get('test_env_provisioning.docker', {})
        
        self.logger.info("Docker environment provisioner initialized")
    
    def validate(self) -> bool:
        """Validate the Docker environment provisioner configuration and state.
        
        Returns:
            True if the provisioner is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        # Check if Docker is available
        try:
            client = docker.from_env()
            client.ping()
        except Exception as e:
            raise ValidationError(f"Docker is not available: {str(e)}")
        
        return True
    
    def provision(self, name: str, datasets: List[str]) -> EnvironmentStatus:
        """Provision a Docker-based test environment.
        
        Args:
            name: Name of the environment.
            datasets: List of dataset names to include.
        
        Returns:
            Status of the provisioned environment.
        
        Raises:
            ProcessingError: If provisioning fails.
        """
        self.logger.info(f"Provisioning Docker environment '{name}' with datasets {datasets}")
        
        # Get Docker parameters
        image = self.docker_config.get('image', 'postgres:latest')
        ports = self.docker_config.get('ports', {'5432/tcp': None})
        environment = self.docker_config.get('environment', {
            'POSTGRES_USER': 'postgres',
            'POSTGRES_PASSWORD': 'postgres',
            'POSTGRES_DB': 'testdb'
        })
        volumes = self.docker_config.get('volumes', [])
        
        # Create environment configuration
        config = EnvironmentConfig(
            name=name,
            type='docker',
            datasets=datasets,
            parameters={
                'image': image,
                'ports': ports,
                'environment': environment,
                'volumes': volumes,
                'db_type': 'postgresql',
                'db_name': environment.get('POSTGRES_DB', 'testdb'),
                'db_user': environment.get('POSTGRES_USER', 'postgres'),
                'db_password': environment.get('POSTGRES_PASSWORD', 'postgres')
            },
            created_at=datetime.now()
        )
        
        # Create environment
        try:
            status = self.environment_manager.create_environment(config)
            return status
        except Exception as e:
            raise ProcessingError(f"Error provisioning Docker environment: {str(e)}")


class FileEnvironmentProvisioner(Component):
    """Component for provisioning file-based test environments."""
    
    def __init__(self, config_manager: ConfigManager, environment_manager: EnvironmentManager):
        """Initialize the file environment provisioner.
        
        Args:
            config_manager: Configuration manager instance.
            environment_manager: EnvironmentManager instance.
        """
        super().__init__(config_manager)
        self.environment_manager = environment_manager
        self.file_config = None
    
    def initialize(self) -> None:
        """Initialize the file environment provisioner.
        
        Raises:
            ConfigurationError: If the provisioner cannot be initialized.
        """
        # Get file configuration
        self.file_config = self.config_manager.get('test_env_provisioning.file', {})
        
        self.logger.info("File environment provisioner initialized")
    
    def validate(self) -> bool:
        """Validate the file environment provisioner configuration and state.
        
        Returns:
            True if the provisioner is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        return True
    
    def provision(self, name: str, datasets: List[str]) -> EnvironmentStatus:
        """Provision a file-based test environment.
        
        Args:
            name: Name of the environment.
            datasets: List of dataset names to include.
        
        Returns:
            Status of the provisioned environment.
        
        Raises:
            ProcessingError: If provisioning fails.
        """
        env_dir = os.path.join(self.output_dir, name)
        os.makedirs(env_dir, exist_ok=True)
        self.logger.info(f"Provisioning file environment '{name}' with datasets {datasets}")
        
        # Get file parameters
        base_dir = self.file_config.get('base_dir')
        
        if not base_dir:
            # Use default base directory
            base_dir = os.path.join(
                self.config_manager.get('general.output_directory', '/output/dwsf'),
                'test_env_provisioning',
                'environments',
                name,
                'data'
            )
        
        # Create environment configuration
        config = EnvironmentConfig(
            name=name,
            type='file',
            datasets=datasets,
            parameters={
                'base_dir': base_dir
            },
            created_at=datetime.now()
        )
        
        # Create environment
        try:
            status = self.environment_manager.create_environment(config)
            return status
        except Exception as e:
            raise ProcessingError(f"Error provisioning file environment: {str(e)}")


class DatabaseEnvironmentProvisioner(Component):
    """Component for provisioning database-based test environments."""
    
    def __init__(self, config_manager: ConfigManager, environment_manager: EnvironmentManager):
        """Initialize the database environment provisioner.
        
        Args:
            config_manager: Configuration manager instance.
            environment_manager: EnvironmentManager instance.
        """
        super().__init__(config_manager)
        self.environment_manager = environment_manager
        self.database_config = None
    
    def initialize(self) -> None:
        """Initialize the database environment provisioner.
        
        Raises:
            ConfigurationError: If the provisioner cannot be initialized.
        """
        # Get database configuration
        self.database_config = self.config_manager.get('test_env_provisioning.database', {})
        
        self.logger.info("Database environment provisioner initialized")
    
    def validate(self) -> bool:
        """Validate the database environment provisioner configuration and state.
        
        Returns:
            True if the provisioner is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        return True
    
    def provision(self, name: str, datasets: List[str]) -> EnvironmentStatus:
        """Provision a database-based test environment.
        
        Args:
            name: Name of the environment.
            datasets: List of dataset names to include.
        
        Returns:
            Status of the provisioned environment.
        
        Raises:
            ProcessingError: If provisioning fails.
        """
        self.logger.info(f"Provisioning database environment '{name}' with datasets {datasets}")
        
        # Get database parameters
        db_type = self.database_config.get('db_type', 'sqlite')
        db_name = self.database_config.get('db_name', 'testdb')
        
        # Create environment configuration
        config = EnvironmentConfig(
            name=name,
            type='database',
            datasets=datasets,
            parameters={
                'db_type': db_type,
                'db_name': db_name
            },
            created_at=datetime.now()
        )
        
        # Create environment
        try:
            status = self.environment_manager.create_environment(config)
            return status
        except Exception as e:
            raise ProcessingError(f"Error provisioning database environment: {str(e)}")


class TestEnvironmentProvisioningPipeline(Pipeline):
    """Pipeline for test environment provisioning."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the test environment provisioning pipeline.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.environment_manager = None
        self.docker_provisioner = None
        self.file_provisioner = None
        self.database_provisioner = None
    
    def initialize(self) -> None:
        """Initialize the pipeline.
        
        Raises:
            ConfigurationError: If the pipeline cannot be initialized.
        """
        # Initialize components
        # self.environment_manager = EnvironmentManager(self.config_manager)
        # self.environment_manager.initialize()
        
        # self.docker_provisioner = DockerEnvironmentProvisioner(self.config_manager, self.environment_manager)
        # self.docker_provisioner.initialize()
        
        self.file_provisioner = FileEnvironmentProvisioner(self.config_manager, self.environment_manager)
        self.file_provisioner.initialize()
        
        self.database_provisioner = DatabaseEnvironmentProvisioner(self.config_manager, self.environment_manager)
        self.database_provisioner.initialize()
        
        # Add pipeline steps
        self.add_step(EnvironmentProvisioningStep(
            self.config_manager,
            self.environment_manager,
            self.docker_provisioner,
            self.file_provisioner,
            self.database_provisioner
        ))
        
        self.logger.info("Test environment provisioning pipeline initialized")
    
    def validate(self) -> bool:
        """Validate the pipeline configuration and state.
        
        Returns:
            True if the pipeline is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        # Validate components
        self.environment_manager.validate()
        
        # Validate provisioners based on configuration
        if self.config_manager.get('test_env_provisioning.docker.enabled', True):
            try:
                self.docker_provisioner.validate()
            except ValidationError as e:
                self.logger.warning(f"Docker provisioner validation failed: {str(e)}")
                self.logger.warning("Docker provisioning will be disabled")
                self.config_manager.config['test_env_provisioning']['docker']['enabled'] = False
        
        self.file_provisioner.validate()
        self.database_provisioner.validate()
        
        return True


class EnvironmentProvisioningStep(PipelineStep):
    """Pipeline step for provisioning test environments."""
    
    def __init__(self, config_manager: ConfigManager,
                environment_manager: EnvironmentManager,
                docker_provisioner: DockerEnvironmentProvisioner,
                file_provisioner: FileEnvironmentProvisioner,
                database_provisioner: DatabaseEnvironmentProvisioner):
        """Initialize the environment provisioning step.
        
        Args:
            config_manager: Configuration manager instance.
            environment_manager: EnvironmentManager instance.
            docker_provisioner: DockerEnvironmentProvisioner instance.
            file_provisioner: FileEnvironmentProvisioner instance.
            database_provisioner: DatabaseEnvironmentProvisioner instance.
        """
        super().__init__(config_manager)
        self.environment_manager = environment_manager
        self.docker_provisioner = docker_provisioner
        self.file_provisioner = file_provisioner
        self.database_provisioner = database_provisioner
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the environment provisioning step.
        
        Args:
            input_data: Dictionary with data, relationships, domain partitions, purpose datasets, and export paths.
        
        Returns:
            Dictionary with input data and environment statuses.
        
        Raises:
            ProcessingError: If provisioning fails.
        """
        self.logger.info("Provisioning test environments")
        
        # Check if provisioning is enabled
        if not self.config_manager.get('test_env_provisioning.enabled', True):
            self.logger.info("Test environment provisioning is disabled in configuration")
            return input_data
        
        # Get provisioning configuration
        provisioning_config = self.config_manager.get('test_env_provisioning.environments', [])
        
        if not provisioning_config:
            self.logger.info("No test environments to provision")
            return input_data
        
        # Get available datasets
        datasets = []
        
        # Add domain partitions
        if 'domain_partitions' in input_data:
            for domain in input_data['domain_partitions'].keys():
                datasets.append(f"domain_{domain}")
        
        # Add purpose-specific datasets
        if 'purpose_datasets' in input_data:
            datasets.extend(input_data['purpose_datasets'].keys())
        
        if not datasets:
            self.logger.warning("No datasets available for provisioning")
            return input_data
        
        # Provision environments
        environment_statuses = []
        
        for env_config in provisioning_config:
            env_name = env_config.get('name')
            env_type = env_config.get('type', 'file')
            env_datasets = env_config.get('datasets', datasets)
            
            if not env_name:
                continue
            
            try:
                # Check if environment already exists
                try:
                    status = self.environment_manager.get_environment_status(env_name)
                    self.logger.info(f"Environment '{env_name}' already exists with status '{status.status}'")
                    environment_statuses.append(status)
                    continue
                except ProcessingError:
                    # Environment doesn't exist, provision it
                    pass
                
                # Provision environment based on type
                if env_type == 'docker' and self.config_manager.get('test_env_provisioning.docker.enabled', True):
                    status = self.docker_provisioner.provision(env_name, env_datasets)
                elif env_type == 'file':
                    status = self.file_provisioner.provision(env_name, env_datasets)
                elif env_type == 'database':
                    status = self.database_provisioner.provision(env_name, env_datasets)
                else:
                    self.logger.warning(f"Unsupported environment type: {env_type}")
                    continue
                
                environment_statuses.append(status)
            except Exception as e:
                self.logger.error(f"Error provisioning environment '{env_name}': {str(e)}")
                # Continue with other environments
        
        self.logger.info(f"Provisioned {len(environment_statuses)} test environments")
        
        return {
            **input_data,
            'environment_statuses': environment_statuses
        }
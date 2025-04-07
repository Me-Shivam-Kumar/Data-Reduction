"""
Data source connectors for the Data Warehouse Subsampling Framework.

This module provides connectors for various data sources, including
databases, files, and APIs.
"""

import os
import logging
import pandas as pd
import numpy as np
import sqlite3
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import csv
from pathlib import Path
import urllib.request
import urllib.parse
import urllib.error
import ssl
import certifi
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .base import Component, ConfigManager, ConfigurationError

logger = logging.getLogger(__name__)


class DataConnector(Component):
    """Base class for data connectors."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the data connector.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.connection = None
    
    def connect(self) -> Any:
        """Establish a connection to the data source.
        
        Returns:
            Connection object.
        
        Raises:
            ConfigurationError: If the connection cannot be established.
        """
        raise NotImplementedError("Subclasses must implement connect()")
    
    def disconnect(self) -> None:
        """Close the connection to the data source."""
        raise NotImplementedError("Subclasses must implement disconnect()")
    
    def get_tables(self) -> List[str]:
        """Get a list of available tables.
        
        Returns:
            List of table names.
        """
        raise NotImplementedError("Subclasses must implement get_tables()")
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get the schema of a table.
        
        Args:
            table_name: Name of the table.
        
        Returns:
            Dictionary mapping column names to data types.
        """
        raise NotImplementedError("Subclasses must implement get_table_schema()")
    
    def get_table_data(self, table_name: str) -> pd.DataFrame:
        """Get the data from a table.
        
        Args:
            table_name: Name of the table.
        
        Returns:
            DataFrame containing the table data.
        """
        raise NotImplementedError("Subclasses must implement get_table_data()")
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute a query.
        
        Args:
            query: SQL query to execute.
            params: Query parameters.
        
        Returns:
            DataFrame containing the query results.
        """
        raise NotImplementedError("Subclasses must implement execute_query()")


class DatabaseConnector(DataConnector):
    """Connector for database data sources."""
    
    def __init__(self, config_manager: ConfigManager, connection_config: Dict[str, Any] = None):
        """Initialize the database connector.
        
        Args:
            config_manager: Configuration manager instance.
            connection_config: Optional connection configuration. If not provided,
                               it will be loaded from the configuration manager.
        """
        super().__init__(config_manager)
        self.connection_config = connection_config or self.config_manager.get('data_sources.primary_warehouse.connection', {})
        self.engine = None
    
    def connect(self) -> Engine:
        """Establish a connection to the database.
        
        Returns:
            SQLAlchemy engine.
        
        Raises:
            ConfigurationError: If the connection cannot be established.
        """
        if self.engine is not None:
            return self.engine
        
        try:
            dialect = self.connection_config.get('dialect', 'sqlite')
            
            if dialect == 'sqlite':
                # SQLite connection
                db_path = self.connection_config.get('path', ':memory:')
                connection_string = f"sqlite:///{db_path}"
            else:
                # Other database connections
                host = self.connection_config.get('host', 'localhost')
                port = self.connection_config.get('port')
                database = self.connection_config.get('database')
                username = self.connection_config.get('username')
                password = self.connection_config.get('password')
                
                if port:
                    connection_string = f"{dialect}://{username}:{password}@{host}:{port}/{database}"
                else:
                    connection_string = f"{dialect}://{username}:{password}@{host}/{database}"
            
            self.engine = create_engine(connection_string)
            self.connection = self.engine.connect()
            
            return self.engine
        except Exception as e:
            raise ConfigurationError(f"Failed to connect to database: {str(e)}")
    
    def disconnect(self) -> None:
        """Close the connection to the database."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
    
    def get_tables(self) -> List[str]:
        """Get a list of available tables.
        
        Returns:
            List of table names.
        """
        engine = self.connect()
        
        if engine.dialect.name == 'sqlite':
            query = "SELECT name FROM sqlite_master WHERE type='table'"
        elif engine.dialect.name == 'postgresql':
            schema = self.connection_config.get('schema', 'public')
            query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'"
        elif engine.dialect.name == 'mysql':
            database = self.connection_config.get('database')
            query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{database}'"
        else:
            raise NotImplementedError(f"get_tables() not implemented for dialect: {engine.dialect.name}")
        
        result = pd.read_sql_query(query, engine)
        return result.iloc[:, 0].tolist()
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get the schema of a table.
        
        Args:
            table_name: Name of the table.
        
        Returns:
            Dictionary mapping column names to data types.
        """
        engine = self.connect()
        
        if engine.dialect.name == 'sqlite':
            query = f"PRAGMA table_info({table_name})"
            result = pd.read_sql_query(query, engine)
            return dict(zip(result['name'], result['type']))
        elif engine.dialect.name == 'postgresql':
            schema = self.connection_config.get('schema', 'public')
            query = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = '{schema}' AND table_name = '{table_name}'
            """
            result = pd.read_sql_query(query, engine)
            return dict(zip(result['column_name'], result['data_type']))
        elif engine.dialect.name == 'mysql':
            database = self.connection_config.get('database')
            query = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = '{database}' AND table_name = '{table_name}'
            """
            result = pd.read_sql_query(query, engine)
            return dict(zip(result['column_name'], result['data_type']))
        else:
            raise NotImplementedError(f"get_table_schema() not implemented for dialect: {engine.dialect.name}")
    
    def get_table_data(self, table_name: str) -> pd.DataFrame:
        """Get the data from a table.
        
        Args:
            table_name: Name of the table.
        
        Returns:
            DataFrame containing the table data.
        """
        engine = self.connect()
        
        # Handle schema if specified
        if engine.dialect.name == 'postgresql':
            schema = self.connection_config.get('schema', 'public')
            full_table_name = f"{schema}.{table_name}"
        else:
            full_table_name = table_name
        
        query = f"SELECT * FROM {full_table_name}"
        return pd.read_sql_query(query, engine)
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute a query.
        
        Args:
            query: SQL query to execute.
            params: Query parameters.
        
        Returns:
            DataFrame containing the query results.
        """
        engine = self.connect()
        
        if params:
            return pd.read_sql_query(text(query), engine, params=params)
        else:
            return pd.read_sql_query(query, engine)


class FileConnector(DataConnector):
    """Connector for file-based data sources."""
    
    def __init__(self, config_manager: ConfigManager, connection_config: Dict[str, Any] = None):
        """Initialize the file connector.
        
        Args:
            config_manager: Configuration manager instance.
            connection_config: Optional connection configuration. If not provided,
                               it will be loaded from the configuration manager.
        """
        super().__init__(config_manager)
        self.connection_config = connection_config or self.config_manager.get('data_sources.primary_warehouse.connection', {})
        self.base_dir = self.connection_config.get('base_dir', '.')
        self.file_format = self.connection_config.get('format', 'csv')
    
    def connect(self) -> str:
        """Establish a connection to the file data source.
        
        Returns:
            Base directory path.
        
        Raises:
            ConfigurationError: If the connection cannot be established.
        """
        if not os.path.exists(self.base_dir):
            raise ConfigurationError(f"Base directory does not exist: {self.base_dir}")
        
        self.connection = self.base_dir
        return self.base_dir
    
    def disconnect(self) -> None:
        """Close the connection to the file data source."""
        self.connection = None
    
    def get_tables(self) -> List[str]:
        """Get a list of available tables (files).
        
        Returns:
            List of table names (file names without extension).
        """
        self.connect()
        
        files = []
        for file in os.listdir(self.base_dir):
            if file.endswith(f".{self.file_format}"):
                files.append(os.path.splitext(file)[0])
        
        return files
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get the schema of a table (file).
        
        Args:
            table_name: Name of the table (file without extension).
        
        Returns:
            Dictionary mapping column names to data types.
        """
        self.connect()
        
        file_path = os.path.join(self.base_dir, f"{table_name}.{self.file_format}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if self.file_format == 'csv':
            df = pd.read_csv(file_path, nrows=1)
        elif self.file_format == 'parquet':
            df = pd.read_parquet(file_path)
        elif self.file_format == 'json':
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
        
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    def get_table_data(self, table_name: str) -> pd.DataFrame:
        """Get the data from a table (file).
        
        Args:
            table_name: Name of the table (file without extension).
        
        Returns:
            DataFrame containing the table data.
        """
        self.connect()
        
        file_path = os.path.join(self.base_dir, f"{table_name}.{self.file_format}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if self.file_format == 'csv':
            return pd.read_csv(file_path)
        elif self.file_format == 'parquet':
            return pd.read_parquet(file_path)
        elif self.file_format == 'json':
            return pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute a query.
        
        This method is not fully supported for file-based connectors.
        It supports a limited subset of SQL-like queries.
        
        Args:
            query: SQL-like query to execute.
            params: Query parameters.
        
        Returns:
            DataFrame containing the query results.
        
        Raises:
            NotImplementedError: If the query is not supported.
        """
        self.connect()
        
        # Very basic SQL-like query support
        query = query.strip().lower()
        
        if query.startswith("select * from "):
            table_name = query[14:].strip()
            return self.get_table_data(table_name)
        else:
            raise NotImplementedError("Only 'SELECT * FROM table' queries are supported for file-based connectors")


class SQLiteConnector(DatabaseConnector):
    """Connector for SQLite databases."""
    
    def __init__(self, config_manager: ConfigManager, connection_config: Dict[str, Any] = None):
        """Initialize the SQLite connector.
        
        Args:
            config_manager: Configuration manager instance.
            connection_config: Optional connection configuration. If not provided,
                               it will be loaded from the configuration manager.
        """
        connection_config = connection_config or config_manager.get('data_sources.primary_warehouse.connection', {})
        connection_config['dialect'] = 'sqlite'
        super().__init__(config_manager, connection_config)


class InMemoryConnector(DataConnector):
    """Connector for in-memory data sources."""
    
    def __init__(self, config_manager: ConfigManager, tables: Dict[str, pd.DataFrame] = None):
        """Initialize the in-memory connector.
        
        Args:
            config_manager: Configuration manager instance.
            tables: Optional dictionary mapping table names to DataFrames.
        """
        super().__init__(config_manager)
        self.tables = tables or {}
    
    def connect(self) -> Dict[str, pd.DataFrame]:
        """Establish a connection to the in-memory data source.
        
        Returns:
            Dictionary mapping table names to DataFrames.
        """
        self.connection = self.tables
        return self.tables
    
    def disconnect(self) -> None:
        """Close the connection to the in-memory data source."""
        self.connection = None
    
    def get_tables(self) -> List[str]:
        """Get a list of available tables.
        
        Returns:
            List of table names.
        """
        self.connect()
        return list(self.tables.keys())
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get the schema of a table.
        
        Args:
            table_name: Name of the table.
        
        Returns:
            Dictionary mapping column names to data types.
        """
        self.connect()
        
        if table_name not in self.tables:
            raise KeyError(f"Table not found: {table_name}")
        
        return {col: str(dtype) for col, dtype in self.tables[table_name].dtypes.items()}
    
    def get_table_data(self, table_name: str) -> pd.DataFrame:
        """Get the data from a table.
        
        Args:
            table_name: Name of the table.
        
        Returns:
            DataFrame containing the table data.
        """
        self.connect()
        
        if table_name not in self.tables:
            raise KeyError(f"Table not found: {table_name}")
        
        return self.tables[table_name].copy()
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute a query.
        
        This method is not fully supported for in-memory connectors.
        It supports a limited subset of SQL-like queries.
        
        Args:
            query: SQL-like query to execute.
            params: Query parameters.
        
        Returns:
            DataFrame containing the query results.
        
        Raises:
            NotImplementedError: If the query is not supported.
        """
        self.connect()
        
        # Very basic SQL-like query support
        query = query.strip().lower()
        
        if query.startswith("select * from "):
            table_name = query[14:].strip()
            return self.get_table_data(table_name)
        else:
            raise NotImplementedError("Only 'SELECT * FROM table' queries are supported for in-memory connectors")


def create_connector(config_manager: ConfigManager) -> DataConnector:
    """Create a data connector based on configuration.
    
    Args:
        config_manager: Configuration manager instance.
    
    Returns:
        Data connector instance.
    
    Raises:
        ConfigurationError: If the connector type is not supported.
    """
    connector_type = config_manager.get('data_sources.primary_warehouse.type', 'database')
    
    if connector_type == 'database':
        dialect = config_manager.get('data_sources.primary_warehouse.connection.dialect', 'sqlite')
        
        if dialect == 'sqlite':
            return SQLiteConnector(config_manager)
        else:
            return DatabaseConnector(config_manager)
    elif connector_type == 'file':
        return FileConnector(config_manager)
    elif connector_type == 'in_memory':
        return InMemoryConnector(config_manager)
    else:
        raise ConfigurationError(f"Unsupported connector type: {connector_type}")

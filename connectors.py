"""
Data connectors for the Data Warehouse Subsampling Framework.

This module provides implementations of data connectors for various data sources,
including relational databases, HDFS, and other storage systems.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import pyodbc
import psycopg2
import pymongo
import pymysql
from sqlalchemy import create_engine, MetaData, Table, Column, inspect
from pyspark.sql import SparkSession

from ..common.base import DataConnector, ConfigManager, ConfigurationError

logger = logging.getLogger(__name__)


class PostgreSQLConnector(DataConnector):
    """Connector for PostgreSQL databases."""
    
    def __init__(self, config_manager: ConfigManager, source_name: str):
        """Initialize the PostgreSQL connector.
        
        Args:
            config_manager: Configuration manager instance.
            source_name: Name of the data source.
        
        Raises:
            ConfigurationError: If the data source configuration is invalid.
        """
        super().__init__(config_manager, source_name)
        self.connection = None
        self.engine = None
    
    def initialize(self) -> None:
        """Initialize the connector.
        
        Raises:
            ConfigurationError: If the connector cannot be initialized.
        """
        try:
            # Create SQLAlchemy engine for metadata operations
            connection_string = self._build_connection_string()
            self.engine = create_engine(connection_string)
            logger.info(f"PostgreSQL connector initialized for source '{self.source_name}'")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize PostgreSQL connector: {str(e)}")
    
    def validate(self) -> bool:
        """Validate the connector configuration and state.
        
        Returns:
            True if the connector is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        try:
            # Test connection
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result is not None and result[0] == 1
        except Exception as e:
            logger.error(f"PostgreSQL connector validation failed: {str(e)}")
            return False
    
    def connect(self) -> Any:
        """Connect to the PostgreSQL database.
        
        Returns:
            Database connection object.
        
        Raises:
            ConnectionError: If connection fails.
        """
        try:
            if self.connection is None or self.connection.closed:
                params = self.source_config.connection_params
                self.connection = psycopg2.connect(
                    host=params.get('host', 'localhost'),
                    port=params.get('port', 5432),
                    database=params.get('database', ''),
                    user=params.get('user', ''),
                    password=params.get('password', ''),
                    sslmode='require' if params.get('ssl', False) else 'prefer'
                )
            return self.connection
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL database: {str(e)}")
    
    def disconnect(self) -> None:
        """Disconnect from the PostgreSQL database."""
        if self.connection and not self.connection.closed:
            self.connection.close()
            self.connection = None
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table.
        
        Args:
            table_name: Name of the table.
        
        Returns:
            Dictionary with schema information.
        """
        schema_name = self.source_config.connection_params.get('schema', 'public')
        
        if '.' in table_name:
            schema_name, table_name = table_name.split('.', 1)
        
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name, schema=schema_name)
            primary_keys = inspector.get_primary_keys(table_name, schema=schema_name)
            foreign_keys = inspector.get_foreign_keys(table_name, schema=schema_name)
            
            return {
                'table_name': table_name,
                'schema_name': schema_name,
                'columns': columns,
                'primary_keys': primary_keys,
                'foreign_keys': foreign_keys
            }
        except Exception as e:
            logger.error(f"Error getting schema for table {schema_name}.{table_name}: {str(e)}")
            return {
                'table_name': table_name,
                'schema_name': schema_name,
                'columns': [],
                'primary_keys': [],
                'foreign_keys': []
            }
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query on the PostgreSQL database.
        
        Args:
            query: Query string.
            params: Query parameters.
        
        Returns:
            Query result.
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or {})
                
                if query.strip().upper().startswith(('SELECT', 'WITH')):
                    columns = [desc[0] for desc in cursor.description]
                    result = cursor.fetchall()
                    return pd.DataFrame(result, columns=columns)
                else:
                    conn.commit()
                    return cursor.rowcount
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def _build_connection_string(self) -> str:
        """Build a connection string for SQLAlchemy.
        
        Returns:
            Connection string.
        """
        params = self.source_config.connection_params
        host = params.get('host', 'localhost')
        port = params.get('port', 5432)
        database = params.get('database', '')
        user = params.get('user', '')
        password = params.get('password', '')
        schema = params.get('schema', 'public')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"


class MySQLConnector(DataConnector):
    """Connector for MySQL databases."""
    
    def __init__(self, config_manager: ConfigManager, source_name: str):
        """Initialize the MySQL connector.
        
        Args:
            config_manager: Configuration manager instance.
            source_name: Name of the data source.
        
        Raises:
            ConfigurationError: If the data source configuration is invalid.
        """
        super().__init__(config_manager, source_name)
        self.connection = None
        self.engine = None
    
    def initialize(self) -> None:
        """Initialize the connector.
        
        Raises:
            ConfigurationError: If the connector cannot be initialized.
        """
        try:
            # Create SQLAlchemy engine for metadata operations
            connection_string = self._build_connection_string()
            self.engine = create_engine(connection_string)
            logger.info(f"MySQL connector initialized for source '{self.source_name}'")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize MySQL connector: {str(e)}")
    
    def validate(self) -> bool:
        """Validate the connector configuration and state.
        
        Returns:
            True if the connector is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        try:
            # Test connection
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result is not None and result[0] == 1
        except Exception as e:
            logger.error(f"MySQL connector validation failed: {str(e)}")
            return False
    
    def connect(self) -> Any:
        """Connect to the MySQL database.
        
        Returns:
            Database connection object.
        
        Raises:
            ConnectionError: If connection fails.
        """
        try:
            if self.connection is None:
                params = self.source_config.connection_params
                self.connection = pymysql.connect(
                    host=params.get('host', 'localhost'),
                    port=params.get('port', 3306),
                    database=params.get('database', ''),
                    user=params.get('user', ''),
                    password=params.get('password', ''),
                    ssl_ca=params.get('ssl_ca') if params.get('ssl', False) else None
                )
            return self.connection
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MySQL database: {str(e)}")
    
    def disconnect(self) -> None:
        """Disconnect from the MySQL database."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table.
        
        Args:
            table_name: Name of the table.
        
        Returns:
            Dictionary with schema information.
        """
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            primary_keys = inspector.get_primary_keys(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)
            
            return {
                'table_name': table_name,
                'columns': columns,
                'primary_keys': primary_keys,
                'foreign_keys': foreign_keys
            }
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {str(e)}")
            return {
                'table_name': table_name,
                'columns': [],
                'primary_keys': [],
                'foreign_keys': []
            }
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query on the MySQL database.
        
        Args:
            query: Query string.
            params: Query parameters.
        
        Returns:
            Query result.
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or {})
                
                if query.strip().upper().startswith(('SELECT', 'WITH')):
                    columns = [desc[0] for desc in cursor.description]
                    result = cursor.fetchall()
                    return pd.DataFrame(result, columns=columns)
                else:
                    conn.commit()
                    return cursor.rowcount
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def _build_connection_string(self) -> str:
        """Build a connection string for SQLAlchemy.
        
        Returns:
            Connection string.
        """
        params = self.source_config.connection_params
        host = params.get('host', 'localhost')
        port = params.get('port', 3306)
        database = params.get('database', '')
        user = params.get('user', '')
        password = params.get('password', '')
        
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"


class SparkConnector(DataConnector):
    """Connector for Apache Spark."""
    
    def __init__(self, config_manager: ConfigManager, source_name: str):
        """Initialize the Spark connector.
        
        Args:
            config_manager: Configuration manager instance.
            source_name: Name of the data source.
        
        Raises:
            ConfigurationError: If the data source configuration is invalid.
        """
        super().__init__(config_manager, source_name)
        self.spark = None
    
    def initialize(self) -> None:
        """Initialize the connector.
        
        Raises:
            ConfigurationError: If the connector cannot be initialized.
        """
        try:
            # Get Spark configuration
            spark_config = self.config_manager.get('data_sources.spark', {})
            
            # Create Spark session
            builder = SparkSession.builder.appName(
                spark_config.get('app_name', 'DataWarehouseSubsampling')
            )
            
            # Set master
            master = spark_config.get('master', 'local[*]')
            builder = builder.master(master)
            
            # Set executor memory
            executor_memory = spark_config.get('executor_memory')
            if executor_memory:
                builder = builder.config('spark.executor.memory', executor_memory)
            
            # Set executor cores
            executor_cores = spark_config.get('executor_cores')
            if executor_cores:
                builder = builder.config('spark.executor.cores', str(executor_cores))
            
            # Set driver memory
            driver_memory = spark_config.get('driver_memory')
            if driver_memory:
                builder = builder.config('spark.driver.memory', driver_memory)
            
            # Set max result size
            max_result_size = spark_config.get('max_result_size')
            if max_result_size:
                builder = builder.config('spark.driver.maxResultSize', max_result_size)
            
            # Add packages
            packages = spark_config.get('packages', [])
            if packages:
                builder = builder.config('spark.jars.packages', ','.join(packages))
            
            # Create Spark session
            self.spark = builder.getOrCreate()
            
            logger.info(f"Spark connector initialized for source '{self.source_name}'")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Spark connector: {str(e)}")
    
    def validate(self) -> bool:
        """Validate the connector configuration and state.
        
        Returns:
            True if the connector is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        try:
            # Test Spark session
            if self.spark is None:
                self.connect()
            
            # Run a simple Spark job
            test_df = self.spark.createDataFrame([(1, 'test')], ['id', 'value'])
            count = test_df.count()
            
            return count == 1
        except Exception as e:
            logger.error(f"Spark connector validation failed: {str(e)}")
            return False
    
    def connect(self) -> Any:
        """Connect to Spark.
        
        Returns:
            Spark session.
        
        Raises:
            ConnectionError: If connection fails.
        """
        try:
            if self.spark is None:
                self.initialize()
            return self.spark
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Spark: {str(e)}")
    
    def disconnect(self) -> None:
        """Disconnect from Spark."""
        if self.spark:
            self.spark.stop()
            self.spark = None
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table.
        
        Args:
            table_name: Name of the table.
        
        Returns:
            Dictionary with schema information.
        """
        try:
            spark = self.connect()
            df = spark.table(table_name)
            schema = df.schema.jsonValue()
            
            return {
                'table_name': table_name,
                'schema': schema
            }
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {str(e)}")
            return {
                'table_name': table_name,
                'schema': {}
            }
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query on Spark.
        
        Args:
            query: SQL query string.
            params: Query parameters (not used for Spark).
        
        Returns:
            Spark DataFrame.
        """
        try:
            spark = self.connect()
            return spark.sql(query)
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise


class MongoDBConnector(DataConnector):
    """Connector for MongoDB databases."""
    
    def __init__(self, config_manager: ConfigManager, source_name: str):
        """Initialize the MongoDB connector.
        
        Args:
            config_manager: Configuration manager instance.
            source_name: Name of the data source.
        
        Raises:
            ConfigurationError: If the data source configuration is invalid.
        """
        super().__init__(config_manager, source_name)
        self.client = None
        self.db = None
    
    def initialize(self) -> None:
        """Initialize the connector.
        
        Raises:
            ConfigurationError: If the connector cannot be initialized.
        """
        try:
            # Connect to MongoDB
            self.connect()
            logger.info(f"MongoDB connector initialized for source '{self.source_name}'")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize MongoDB connector: {str(e)}")
    
    def validate(self) -> bool:
        """Validate the connector configuration and state.
        
        Returns:
            True if the connector is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        try:
            # Test connection
            client = self.connect()
            server_info = client.server_info()
            return 'version' in server_info
        except Exception as e:
            logger.error(f"MongoDB connector validation failed: {str(e)}")
            return False
    
    def connect(self) -> Any:
        """Connect to the MongoDB database.
        
        Returns:
            MongoDB client.
        
        Raises:
            ConnectionError: If connection fails.
        """
        try:
            if self.client is None:
                params = self.source_config.connection_params
                host = params.get('host', 'localhost')
                port = params.get('port', 27017)
                database = params.get('database', '')
                user = params.get('user', '')
                password = params.get('password', '')
                
                # Build connection string
                if user and password:
                    connection_string = f"mongodb://{user}:{password}@{host}:{port}/{database}"
                else:
                    connection_string = f"mongodb://{host}:{port}/{database}"
                
                # Connect to MongoDB
                self.client = pymongo.MongoClient(connection_string)
                self.db = self.client[database]
            
            return self.client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")
    
    def disconnect(self) -> None:
        """Disconnect from the MongoDB database."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a collection.
        
        Args:
            table_name: Name of the collection.
        
        Returns:
            Dictionary with schema information.
        """
        try:
            # Connect to MongoDB
            self.connect()
            
            # Get collection
            collection = self.db[table_name]
            
            # Infer schema from a sample document
            sample_document = collection.find_one()
            
            if sample_document:
                # Remove _id field
                if '_id' in sample_document:
                    del sample_document['_id']
                
                # Infer schema
                schema = {
                    'fields': []
                }
                
                for field, value in sample_document.items():
                    field_type = type(value).__name__
                    schema['fields'].append({
                        'name': field,
                        'type': field_type
                    })
                
                return {
                    'collection_name': table_name,
                    'schema': schema
                }
            else:
                return {
                    'collection_name': table_name,
                    'schema': {'fields': []}
                }
        except Exception as e:
            logger.error(f"Error getting schema for collection {table_name}: {str(e)}")
            return {
                'collection_name': table_name,
                'schema': {'fields': []}
            }
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query on the MongoDB database.
        
        Args:
            query: Query string (not used for MongoDB).
            params: Query parameters.
        
        Returns:
            Query result.
        """
        try:
            # Connect to MongoDB
            self.connect()
            
            # Parse params
            if not params:
                raise ValueError("MongoDB queries require parameters")
            
            collection_name = params.get('collection')
            if not collection_name:
                raise ValueError("Collection name is required")
            
            operation = params.get('operation', 'find')
            filter_dict = params.get('filter', {})
            projection = params.get('projection', None)
            sort = params.get('sort', None)
            limit = params.get('limit', None)
            
            # Get collection
            collection = self.db[collection_name]
            
            # Execute operation
            if operation == 'find':
                cursor = collection.find(filter_dict, projection)
                
                if sort:
                    cursor = cursor.sort(sort)
                
                if limit:
                    cursor = cursor.limit(limit)
                
                # Convert to DataFrame
                result = list(cursor)
                
                # Remove _id field
                for doc in result:
                    if '_id' in doc:
                        del doc['_id']
                
                return pd.DataFrame(result)
            elif operation == 'insert_one':
                document = params.get('document', {})
                result = collection.insert_one(document)
                return {'inserted_id': str(result.inserted_id)}
            elif operation == 'insert_many':
                documents = params.get('documents', [])
                result = collection.insert_many(documents)
                return {'inserted_ids': [str(id) for id in result.inserted_ids]}
            elif operation == 'update_one':
                update = params.get('update', {})
                result = collection.update_one(filter_dict, update)
                return {
                    'matched_count': result.matched_count,
                    'modified_count': result.modified_count
                }
            elif operation == 'update_many':
                update = params.get('update', {})
                result = collection.update_many(filter_dict, update)
                return {
                    'matched_count': result.matched_count,
                    'modified_count': result.modified_count
                }
            elif operation == 'delete_one':
                result = collection.delete_one(filter_dict)
                return {'deleted_count': result.deleted_count}
            elif operation == 'delete_many':
                result = collection.delete_many(filter_dict)
                return {'deleted_count': result.deleted_count}
            elif operation == 'count':
                return {'count': collection.count_documents(filter_dict)}
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.error(f"Error executing MongoDB query: {str(e)}")
            raise


# Factory function to create the appropriate connector
def create_connector(config_manager: ConfigManager, source_name: str) -> DataConnector:
    """Create a data connector for the specified data source.
    
    Args:
        config_manager: Configuration manager instance.
        source_name: Name of the data source.
    
    Returns:
        Data connector instance.
    
    Raises:
        ConfigurationError: If the data source configuration is invalid.
    """
    source_config = config_manager.get_data_source_config(source_name)
    
    if not source_config:
        raise ConfigurationError(f"Data source '{source_name}' not found in configuration")
    
    source_type = source_config.type.lower()
    
    if source_type == 'postgresql':
        return PostgreSQLConnector(config_manager, source_name)
    elif source_type == 'mysql':
        return MySQLConnector(config_manager, source_name)
    elif source_type == 'spark':
        return SparkConnector(config_manager, source_name)
    elif source_type == 'mongodb':
        return MongoDBConnector(config_manager, source_name)
    else:
        raise ConfigurationError(f"Unsupported data source type: {source_type}")

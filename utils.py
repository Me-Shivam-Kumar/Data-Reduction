"""
Utility functions for the Data Warehouse Subsampling Framework.

This module provides utility functions used throughout the framework.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import hashlib
import datetime
import csv
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_directory(directory: str) -> str:
    """Ensure a directory exists.
    
    Args:
        directory: Directory path.
    
    Returns:
        The directory path.
    """
    os.makedirs(directory, exist_ok=True)
    return directory


def save_dataframe(df: pd.DataFrame, file_path: str, format: str = 'csv') -> str:
    """Save a DataFrame to a file.
    
    Args:
        df: DataFrame to save.
        file_path: Path to save the DataFrame to.
        format: File format (csv, parquet, json).
    
    Returns:
        The file path.
    """
    directory = os.path.dirname(file_path)
    ensure_directory(directory)
    
    if format == 'csv':
        df.to_csv(file_path, index=False)
    elif format == 'parquet':
        df.to_parquet(file_path, index=False)
    elif format == 'json':
        df.to_json(file_path, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return file_path


def load_dataframe(file_path: str, format: str = None) -> pd.DataFrame:
    """Load a DataFrame from a file.
    
    Args:
        file_path: Path to load the DataFrame from.
        format: File format (csv, parquet, json). If None, inferred from file extension.
    
    Returns:
        The loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if format is None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            format = 'csv'
        elif ext == '.parquet':
            format = 'parquet'
        elif ext == '.json':
            format = 'json'
        else:
            raise ValueError(f"Could not infer format from file extension: {ext}")
    
    if format == 'csv':
        return pd.read_csv(file_path)
    elif format == 'parquet':
        return pd.read_parquet(file_path)
    elif format == 'json':
        return pd.read_json(file_path, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_json(data: Any, file_path: str) -> str:
    """Save data to a JSON file.
    
    Args:
        data: Data to save.
        file_path: Path to save the data to.
    
    Returns:
        The file path.
    """
    directory = os.path.dirname(file_path)
    ensure_directory(directory)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=json_serializer)
    
    return file_path


def load_json(file_path: str) -> Any:
    """Load data from a JSON file.
    
    Args:
        file_path: Path to load the data from.
    
    Returns:
        The loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def json_serializer(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code.
    
    Args:
        obj: Object to serialize.
    
    Returns:
        Serialized object.
    """
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    
    raise TypeError(f"Type {type(obj)} not serializable")


def hash_dataframe(df: pd.DataFrame) -> str:
    """Compute a hash of a DataFrame.
    
    Args:
        df: DataFrame to hash.
    
    Returns:
        Hash of the DataFrame.
    """
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()


def create_sqlite_database(db_path: str, tables: Dict[str, pd.DataFrame]) -> str:
    """Create a SQLite database from DataFrames.
    
    Args:
        db_path: Path to the SQLite database file.
        tables: Dictionary mapping table names to DataFrames.
    
    Returns:
        The database path.
    """
    directory = os.path.dirname(db_path)
    ensure_directory(directory)
    
    conn = sqlite3.connect(db_path)
    
    for table_name, df in tables.items():
        df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    conn.close()
    
    return db_path


def query_sqlite_database(db_path: str, query: str) -> pd.DataFrame:
    """Query a SQLite database.
    
    Args:
        db_path: Path to the SQLite database file.
        query: SQL query to execute.
    
    Returns:
        The query result as a DataFrame.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    result = pd.read_sql_query(query, conn)
    conn.close()
    
    return result


def get_file_size(file_path: str) -> int:
    """Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file.
    
    Returns:
        Size of the file in bytes.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return os.path.getsize(file_path)


def format_file_size(size_bytes: int) -> str:
    """Format a file size in bytes to a human-readable string.
    
    Args:
        size_bytes: Size in bytes.
    
    Returns:
        Human-readable string.
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_directory_size(directory: str) -> int:
    """Get the total size of a directory in bytes.
    
    Args:
        directory: Path to the directory.
    
    Returns:
        Total size of the directory in bytes.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    
    return total_size


def list_files(directory: str, pattern: str = None) -> List[str]:
    """List files in a directory.
    
    Args:
        directory: Path to the directory.
        pattern: Glob pattern to filter files.
    
    Returns:
        List of file paths.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if pattern:
        return [str(p) for p in Path(directory).glob(pattern)]
    else:
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def create_csv_file(file_path: str, headers: List[str], rows: List[List[Any]]) -> str:
    """Create a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        headers: List of column headers.
        rows: List of rows, where each row is a list of values.
    
    Returns:
        The file path.
    """
    directory = os.path.dirname(file_path)
    ensure_directory(directory)
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    return file_path


def read_csv_file(file_path: str) -> Tuple[List[str], List[List[Any]]]:
    """Read a CSV file.
    
    Args:
        file_path: Path to the CSV file.
    
    Returns:
        Tuple of (headers, rows).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
    
    return headers, rows

"""
Utility functions for the Data Warehouse Subsampling Framework.

This module provides common utility functions used across the framework,
including data manipulation, validation, and helper functions.
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_data_frame(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """Validate that a DataFrame contains the required columns.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must be present.
    
    Returns:
        True if the DataFrame is valid, False otherwise.
    """
    if df is None or df.empty:
        logger.warning("DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"DataFrame is missing required columns: {missing_columns}")
            return False
    
    return True


def calculate_data_reduction_ratio(original_size: int, reduced_size: int) -> float:
    """Calculate the data reduction ratio.
    
    Args:
        original_size: Original data size in bytes.
        reduced_size: Reduced data size in bytes.
    
    Returns:
        Reduction ratio (original_size / reduced_size).
    """
    if reduced_size <= 0:
        return float('inf')
    
    return original_size / reduced_size


def calculate_anomaly_preservation_rate(original_anomalies: int, preserved_anomalies: int) -> float:
    """Calculate the anomaly preservation rate.
    
    Args:
        original_anomalies: Number of anomalies in the original data.
        preserved_anomalies: Number of anomalies preserved in the sampled data.
    
    Returns:
        Preservation rate as a percentage.
    """
    if original_anomalies <= 0:
        return 100.0
    
    return (preserved_anomalies / original_anomalies) * 100.0


def get_file_size(file_path: str) -> int:
    """Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file.
    
    Returns:
        File size in bytes.
    """
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError) as e:
        logger.error(f"Error getting file size for {file_path}: {str(e)}")
        return 0


def get_directory_size(directory_path: str) -> int:
    """Get the total size of a directory in bytes.
    
    Args:
        directory_path: Path to the directory.
    
    Returns:
        Directory size in bytes.
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += get_file_size(file_path)
        return total_size
    except (OSError, FileNotFoundError) as e:
        logger.error(f"Error getting directory size for {directory_path}: {str(e)}")
        return 0


def calculate_hash(data: Any) -> str:
    """Calculate a hash for data.
    
    Args:
        data: Data to hash.
    
    Returns:
        Hash string.
    """
    if isinstance(data, pd.DataFrame):
        data_str = data.to_json(orient='records')
    elif isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    
    return hashlib.sha256(data_str.encode()).hexdigest()


def split_dataframe_by_domain(df: pd.DataFrame, domain_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Split a DataFrame into multiple DataFrames based on domain configuration.
    
    Args:
        df: Input DataFrame.
        domain_config: Domain configuration dictionary.
    
    Returns:
        Dictionary mapping domain names to DataFrames.
    """
    result = {}
    
    for domain in domain_config.get('domains', []):
        domain_name = domain.get('name')
        domain_tables = domain.get('tables', [])
        
        if 'table_column' in domain:
            # Split based on a column value
            column_name = domain.get('table_column')
            column_values = domain.get('values', [])
            
            if column_name in df.columns:
                domain_df = df[df[column_name].isin(column_values)]
                result[domain_name] = domain_df
        elif 'table' in domain:
            # Filter based on table name
            table_name = domain.get('table')
            if 'table_name' in df.columns and table_name in df['table_name'].unique():
                domain_df = df[df['table_name'] == table_name]
                result[domain_name] = domain_df
    
    return result


def detect_relationships(dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Detect potential relationships between DataFrames.
    
    Args:
        dfs: Dictionary mapping table names to DataFrames.
    
    Returns:
        List of dictionaries describing detected relationships.
    """
    relationships = []
    
    # Get all column names across all DataFrames
    all_columns = {}
    for table_name, df in dfs.items():
        all_columns[table_name] = set(df.columns)
    
    # Look for potential foreign key relationships
    for table1, columns1 in all_columns.items():
        for table2, columns2 in all_columns.items():
            if table1 == table2:
                continue
            
            # Look for columns with the same name
            common_columns = columns1.intersection(columns2)
            for column in common_columns:
                # Check if values in table1[column] are a subset of table2[column]
                values1 = set(dfs[table1][column].dropna().unique())
                values2 = set(dfs[table2][column].dropna().unique())
                
                if values1.issubset(values2) and len(values1) > 0:
                    relationships.append({
                        'parent_table': table2,
                        'parent_column': column,
                        'child_table': table1,
                        'child_column': column,
                        'confidence': len(values1) / len(values2) if len(values2) > 0 else 0
                    })
    
    return relationships


def stratified_sampling(df: pd.DataFrame, strata_columns: List[str], sample_sizes: Dict[str, float]) -> pd.DataFrame:
    """Perform stratified sampling on a DataFrame.
    
    Args:
        df: Input DataFrame.
        strata_columns: Columns to stratify by.
        sample_sizes: Dictionary mapping strata values to sample sizes.
    
    Returns:
        Sampled DataFrame.
    """
    if not validate_data_frame(df, strata_columns):
        logger.warning("Invalid DataFrame for stratified sampling")
        return df
    
    result = pd.DataFrame()
    
    # Group by strata columns
    grouped = df.groupby(strata_columns)
    
    # Sample each group
    for name, group in grouped:
        # Determine sample size for this group
        strata_key = '_'.join(str(x) for x in name) if isinstance(name, tuple) else str(name)
        sample_size = sample_sizes.get(strata_key, sample_sizes.get('default', 0.1))
        
        # Calculate number of samples
        n_samples = max(1, int(len(group) * sample_size))
        
        # Sample the group
        sampled_group = group.sample(n=n_samples, random_state=42)
        
        # Append to result
        result = pd.concat([result, sampled_group])
    
    return result


def entity_based_subsetting(df: pd.DataFrame, entity_config: Dict[str, Any], 
                           related_dfs: Dict[str, pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Perform entity-based subsetting on a DataFrame and related DataFrames.
    
    Args:
        df: Input DataFrame containing the primary entities.
        entity_config: Configuration for entity-based subsetting.
        related_dfs: Dictionary of related DataFrames to subset based on the selected entities.
    
    Returns:
        Tuple of (sampled primary DataFrame, dictionary of sampled related DataFrames).
    """
    if not validate_data_frame(df):
        logger.warning("Invalid DataFrame for entity-based subsetting")
        return df, related_dfs or {}
    
    # Get entity ID column
    id_column = entity_config.get('id_column')
    if id_column not in df.columns:
        logger.warning(f"Entity ID column '{id_column}' not found in DataFrame")
        return df, related_dfs or {}
    
    # Determine sample size
    sample_size = entity_config.get('sample_size', 0.1)
    
    # Select entities based on the specified method
    selection_method = entity_config.get('selection_method', 'random')
    if selection_method == 'stratified' and 'strata_columns' in entity_config:
        strata_columns = entity_config.get('strata_columns', [])
        strata_sample_sizes = entity_config.get('strata_sample_sizes', {'default': sample_size})
        sampled_df = stratified_sampling(df, strata_columns, strata_sample_sizes)
    elif selection_method == 'systematic':
        # Systematic sampling
        n_samples = max(1, int(len(df) * sample_size))
        step = max(1, len(df) // n_samples)
        sampled_df = df.iloc[::step].head(n_samples)
    else:
        # Random sampling
        n_samples = max(1, int(len(df) * sample_size))
        sampled_df = df.sample(n=n_samples, random_state=42)
    
    # Get the selected entity IDs
    selected_entity_ids = set(sampled_df[id_column].unique())
    
    # Subset related DataFrames based on the selected entity IDs
    sampled_related_dfs = {}
    if related_dfs:
        for table_name, related_df in related_dfs.items():
            relation_config = entity_config.get('relations', {}).get(table_name, {})
            foreign_key = relation_config.get('foreign_key', id_column)
            
            if foreign_key in related_df.columns:
                # Filter related DataFrame to only include rows related to selected entities
                sampled_related_df = related_df[related_df[foreign_key].isin(selected_entity_ids)]
                sampled_related_dfs[table_name] = sampled_related_df
            else:
                logger.warning(f"Foreign key column '{foreign_key}' not found in related DataFrame '{table_name}'")
                sampled_related_dfs[table_name] = related_df
    
    return sampled_df, sampled_related_dfs


def boundary_value_extraction(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Extract boundary values from a DataFrame.
    
    Args:
        df: Input DataFrame.
        config: Configuration for boundary value extraction.
    
    Returns:
        DataFrame containing boundary values.
    """
    if not validate_data_frame(df):
        logger.warning("Invalid DataFrame for boundary value extraction")
        return df
    
    result = pd.DataFrame()
    
    # Get numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    for column in numeric_columns:
        column_values = df[column].dropna()
        
        if len(column_values) == 0:
            continue
        
        boundary_rows = pd.DataFrame()
        
        # Include min and max values
        if config.get('min_max', True):
            min_value = column_values.min()
            max_value = column_values.max()
            
            min_rows = df[df[column] == min_value]
            max_rows = df[df[column] == max_value]
            
            boundary_rows = pd.concat([boundary_rows, min_rows, max_rows])
        
        # Include percentile values
        if 'percentiles' in config:
            percentiles = config.get('percentiles', [])
            for p in percentiles:
                percentile_value = np.percentile(column_values, p)
                # Find closest value in the DataFrame
                closest_idx = (df[column] - percentile_value).abs().argsort()[:5]
                percentile_rows = df.iloc[closest_idx]
                boundary_rows = pd.concat([boundary_rows, percentile_rows])
        
        # Include outliers
        if config.get('outliers', False):
            q1 = column_values.quantile(0.25)
            q3 = column_values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_rows = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            boundary_rows = pd.concat([boundary_rows, outlier_rows])
        
        # Remove duplicates and add to result
        boundary_rows = boundary_rows.drop_duplicates()
        result = pd.concat([result, boundary_rows])
    
    # Remove duplicates in the final result
    result = result.drop_duplicates()
    
    return result


def preserve_referential_integrity(main_df: pd.DataFrame, related_dfs: Dict[str, pd.DataFrame], 
                                  relationships: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """Preserve referential integrity between DataFrames.
    
    Args:
        main_df: Main DataFrame.
        related_dfs: Dictionary of related DataFrames.
        relationships: List of relationship dictionaries.
    
    Returns:
        Dictionary of DataFrames with referential integrity preserved.
    """
    result = {
        'main': main_df.copy()
    }
    
    for table_name, df in related_dfs.items():
        result[table_name] = df.copy()
    
    # Process each relationship
    for relationship in relationships:
        parent_table = relationship.get('parent_table')
        parent_column = relationship.get('parent_column')
        child_table = relationship.get('child_table')
        child_column = relationship.get('child_column')
        
        # Skip if any table is missing
        if parent_table not in result and parent_table != 'main':
            continue
        if child_table not in result and child_table != 'main':
            continue
        
        # Get the DataFrames
        parent_df = result['main'] if parent_table == 'main' else result[parent_table]
        child_df = result['main'] if child_table == 'main' else result[child_table]
        
        # Skip if any column is missing
        if parent_column not in parent_df.columns or child_column not in child_df.columns:
            continue
        
        # Get the values in the child DataFrame
        child_values = set(child_df[child_column].dropna().unique())
        
        # Get the values in the parent DataFrame
        parent_values = set(parent_df[parent_column].dropna().unique())
        
        # Find values in the child that are not in the parent
        missing_values = child_values - parent_values
        
        if missing_values:
            # Filter rows from the parent DataFrame that have the missing values
            missing_rows = parent_df[parent_df[parent_column].isin(missing_values)]
            
            # Add the missing rows to the parent DataFrame
            if not missing_rows.empty:
                result[parent_table] = pd.concat([parent_df, missing_rows])
    
    return result

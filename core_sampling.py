"""
Core Sampling module for the Data Warehouse Subsampling Framework.

This module provides components for sampling data using various techniques
to reduce data volume while maintaining statistical validity and preserving
important characteristics.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import json
from collections import defaultdict

from ..common.base import Component, ConfigManager, Pipeline, PipelineStep, SamplingResult, ProcessingError
from ..common.utils import save_dataframe, load_dataframe, save_json, load_json, ensure_directory

logger = logging.getLogger(__name__)


class Sampler(Component):
    """Base class for data samplers."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the sampler.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
    
    def sample(self, df: pd.DataFrame, table_name: str, domain: str) -> Tuple[pd.DataFrame, SamplingResult]:
        """Sample a DataFrame.
        
        Args:
            df: DataFrame to sample.
            table_name: Name of the table.
            domain: Domain of the table.
        
        Returns:
            Tuple of (sampled DataFrame, sampling result).
        """
        raise NotImplementedError("Subclasses must implement sample()")


class RandomSampler(Sampler):
    """Sampler using random sampling."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the random sampler.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.default_sample_size = None
        self.table_sample_sizes = None
    
    def initialize(self) -> None:
        """Initialize the random sampler.
        
        Raises:
            ConfigurationError: If the sampler cannot be initialized.
        """
        self.default_sample_size = self.config_manager.get('core_sampling.methods.random.default_sample_size', 0.1)
        self.table_sample_sizes = self.config_manager.get('core_sampling.methods.random.table_sample_sizes', {})
        
        self.logger.info(f"Random sampler initialized with default sample size: {self.default_sample_size}")
    
    def sample(self, df: pd.DataFrame, table_name: str, domain: str) -> Tuple[pd.DataFrame, SamplingResult]:
        """Sample a DataFrame using random sampling.
        
        Args:
            df: DataFrame to sample.
            table_name: Name of the table.
            domain: Domain of the table.
        
        Returns:
            Tuple of (sampled DataFrame, sampling result).
        """
        self.logger.info(f"Sampling table {table_name} using random sampling")
        
        # Get sample size for this table
        sample_size = self.table_sample_sizes.get(table_name, self.default_sample_size)
        
        # Calculate number of rows to sample
        if isinstance(sample_size, float) and 0 < sample_size < 1:
            n_samples = int(len(df) * sample_size)
        else:
            n_samples = int(sample_size)
        
        # Ensure at least one row is sampled
        n_samples = max(1, min(n_samples, len(df)))
        
        # Sample data
        sampled_df = df.sample(n=n_samples, random_state=42)
        
        # Create sampling result
        result = SamplingResult(
            table_name=table_name,
            domain=domain,
            original_row_count=len(df),
            sampled_row_count=len(sampled_df),
            sampling_method='random',
            sampling_rate=len(sampled_df) / len(df) if len(df) > 0 else 0.0,
            metadata={
                'sample_size': sample_size
            }
        )
        
        self.logger.info(f"Sampled {len(sampled_df)} rows from table {table_name} (reduction: {100 * (1 - result.sampling_rate):.2f}%)")
        return sampled_df, result


class StratifiedSampler(Sampler):
    """Sampler using stratified sampling."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the stratified sampler.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.default_sample_size = None
        self.strata_config = None
    
    def initialize(self) -> None:
        """Initialize the stratified sampler.
        
        Raises:
            ConfigurationError: If the sampler cannot be initialized.
        """
        self.default_sample_size = self.config_manager.get('core_sampling.methods.stratified.default_sample_size', 0.1)
        self.strata_config = self.config_manager.get('core_sampling.methods.stratified.strata', {})
        
        self.logger.info(f"Stratified sampler initialized with default sample size: {self.default_sample_size}")
    
    def sample(self, df: pd.DataFrame, table_name: str, domain: str) -> Tuple[pd.DataFrame, SamplingResult]:
        """Sample a DataFrame using stratified sampling.
        
        Args:
            df: DataFrame to sample.
            table_name: Name of the table.
            domain: Domain of the table.
        
        Returns:
            Tuple of (sampled DataFrame, sampling result).
        """
        self.logger.info(f"Sampling table {table_name} using stratified sampling")
        
        # Get strata configuration for this table
        table_config = self.strata_config.get(table_name, {})
        strata_columns = table_config.get('strata_columns', [])
        strata_sample_sizes = table_config.get('strata_sample_sizes', {})
        default_strata_sample_size = strata_sample_sizes.get('default', self.default_sample_size)
        
        # If no strata columns specified, fall back to random sampling
        if not strata_columns or not all(col in df.columns for col in strata_columns):
            self.logger.warning(f"No valid strata columns specified for table {table_name}, falling back to random sampling")
            return RandomSampler(self.config_manager).sample(df, table_name, domain)
        
        # Group data by strata
        grouped = df.groupby(strata_columns)
        
        # Sample from each stratum
        sampled_dfs = []
        
        for name, group in grouped:
            # Get sample size for this stratum
            stratum_key = '_'.join(str(n) for n in name) if isinstance(name, tuple) else str(name)
            sample_size = strata_sample_sizes.get(stratum_key, default_strata_sample_size)
            
            # Calculate number of rows to sample
            if isinstance(sample_size, float) and 0 < sample_size < 1:
                n_samples = int(len(group) * sample_size)
            else:
                n_samples = int(sample_size)
            
            # Ensure at least one row is sampled
            n_samples = max(1, min(n_samples, len(group)))
            
            # Sample data
            sampled_group = group.sample(n=n_samples, random_state=42)
            sampled_dfs.append(sampled_group)
        
        # Combine sampled data
        sampled_df = pd.concat(sampled_dfs) if sampled_dfs else pd.DataFrame(columns=df.columns)
        
        # Create sampling result
        result = SamplingResult(
            table_name=table_name,
            domain=domain,
            original_row_count=len(df),
            sampled_row_count=len(sampled_df),
            sampling_method='stratified',
            sampling_rate=len(sampled_df) / len(df) if len(df) > 0 else 0.0,
            metadata={
                'strata_columns': strata_columns,
                'strata_counts': {str(name): len(group) for name, group in grouped},
                'sampled_strata_counts': {str(name): len(group) for name, group in sampled_df.groupby(strata_columns)}
            }
        )
        
        self.logger.info(f"Sampled {len(sampled_df)} rows from table {table_name} (reduction: {100 * (1 - result.sampling_rate):.2f}%)")
        return sampled_df, result


class EntityBasedSampler(Sampler):
    """Sampler using entity-based sampling."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the entity-based sampler.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.entities_config = None
        self.relationships = None
    
    def initialize(self) -> None:
        """Initialize the entity-based sampler.
        
        Raises:
            ConfigurationError: If the sampler cannot be initialized.
        """
        self.entities_config = self.config_manager.get('core_sampling.methods.entity_based.entities', {})
        
        # Relationships will be loaded from data
        
        self.logger.info(f"Entity-based sampler initialized with {len(self.entities_config)} entities")
    
    def sample(self, df: pd.DataFrame, table_name: str, domain: str) -> Tuple[pd.DataFrame, SamplingResult]:
        """Sample a DataFrame using entity-based sampling.
        
        This method is not meant to be called directly, as entity-based sampling
        operates on multiple tables. It falls back to random sampling.
        
        Args:
            df: DataFrame to sample.
            table_name: Name of the table.
            domain: Domain of the table.
        
        Returns:
            Tuple of (sampled DataFrame, sampling result).
        """
        self.logger.warning(f"Entity-based sampling called directly on table {table_name}, falling back to random sampling")
        return RandomSampler(self.config_manager).sample(df, table_name, domain)
    
    def sample_entity(self, tables: Dict[str, pd.DataFrame], entity_name: str, relationships: List[Dict[str, Any]]) -> Dict[str, Tuple[pd.DataFrame, SamplingResult]]:
        """Sample tables based on an entity.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
            entity_name: Name of the entity to sample.
            relationships: List of relationship dictionaries.
        
        Returns:
            Dictionary mapping table names to tuples of (sampled DataFrame, sampling result).
        """
        self.logger.info(f"Sampling entity: {entity_name}")
        
        # Get entity configuration
        entity_config = self.entities_config.get(entity_name, {})
        
        if not entity_config:
            self.logger.warning(f"No configuration found for entity {entity_name}")
            return {}
        
        # Get primary entity table
        primary_entity_table = entity_config.get('primary_entity_table')
        
        if not primary_entity_table or primary_entity_table not in tables:
            self.logger.warning(f"Primary entity table {primary_entity_table} not found for entity {entity_name}")
            return {}
        
        # Get entity ID column
        id_column = entity_config.get('entity_config', {}).get('id_column')
        
        if not id_column or id_column not in tables[primary_entity_table].columns:
            self.logger.warning(f"ID column {id_column} not found in primary entity table {primary_entity_table}")
            return {}
        
        # Get sample size
        sample_size = entity_config.get('entity_config', {}).get('sample_size', 0.1)
        
        # Sample primary entity table
        primary_df = tables[primary_entity_table]
        
        # Calculate number of entities to sample
        if isinstance(sample_size, float) and 0 < sample_size < 1:
            n_samples = int(len(primary_df) * sample_size)
        else:
            n_samples = int(sample_size)
        
        # Ensure at least one entity is sampled
        n_samples = max(1, min(n_samples, len(primary_df)))
        
        # Sample entities
        sampled_entities = primary_df.sample(n=n_samples, random_state=42)
        sampled_entity_ids = set(sampled_entities[id_column])
        
        # Create result for primary entity table
        results = {}
        results[primary_entity_table] = (
            sampled_entities,
            SamplingResult(
                table_name=primary_entity_table,
                domain=entity_name,
                original_row_count=len(primary_df),
                sampled_row_count=len(sampled_entities),
                sampling_method='entity_based',
                sampling_rate=len(sampled_entities) / len(primary_df) if len(primary_df) > 0 else 0.0,
                metadata={
                    'entity_name': entity_name,
                    'is_primary': True,
                    'id_column': id_column,
                    'sample_size': sample_size
                }
            )
        )
        
        # Get related tables
        related_tables = entity_config.get('related_tables', [])
        
        # Sample related tables
        for related_config in related_tables:
            related_table = related_config.get('table')
            join_column = related_config.get('join_column')
            
            if not related_table or related_table not in tables:
                self.logger.warning(f"Related table {related_table} not found for entity {entity_name}")
                continue
            
            if not join_column or join_column not in tables[related_table].columns:
                self.logger.warning(f"Join column {join_column} not found in related table {related_table}")
                continue
            
            # Sample related table based on sampled entity IDs
            related_df = tables[related_table]
            sampled_related = related_df[related_df[join_column].isin(sampled_entity_ids)]
            
            # Create result for related table
            results[related_table] = (
                sampled_related,
                SamplingResult(
                    table_name=related_table,
                    domain=entity_name,
                    original_row_count=len(related_df),
                    sampled_row_count=len(sampled_related),
                    sampling_method='entity_based',
                    sampling_rate=len(sampled_related) / len(related_df) if len(related_df) > 0 else 0.0,
                    metadata={
                        'entity_name': entity_name,
                        'is_primary': False,
                        'join_column': join_column,
                        'primary_table': primary_entity_table,
                        'primary_column': id_column
                    }
                )
            )
        
        # Log summary
        total_original = sum(len(tables[table]) for table in [primary_entity_table] + [r.get('table') for r in related_tables if r.get('table') in tables])
        total_sampled = sum(len(result[0]) for result in results.values())
        
        self.logger.info(f"Sampled entity {entity_name}: {total_sampled} rows from {total_original} rows (reduction: {100 * (1 - total_sampled / total_original):.2f}%)")
        
        return results


class BoundaryValueSampler(Sampler):
    """Sampler using boundary value sampling."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the boundary value sampler.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.default_sample_size = None
        self.boundary_config = None
    
    def initialize(self) -> None:
        """Initialize the boundary value sampler.
        
        Raises:
            ConfigurationError: If the sampler cannot be initialized.
        """
        self.default_sample_size = self.config_manager.get('core_sampling.methods.boundary_value.default_sample_size', 0.1)
        self.boundary_config = self.config_manager.get('core_sampling.methods.boundary_value.boundaries', {})
        
        self.logger.info(f"Boundary value sampler initialized with default sample size: {self.default_sample_size}")
    
    def sample(self, df: pd.DataFrame, table_name: str, domain: str) -> Tuple[pd.DataFrame, SamplingResult]:
        """Sample a DataFrame using boundary value sampling.
        
        Args:
            df: DataFrame to sample.
            table_name: Name of the table.
            domain: Domain of the table.
        
        Returns:
            Tuple of (sampled DataFrame, sampling result).
        """
        self.logger.info(f"Sampling table {table_name} using boundary value sampling")
        
        # Get boundary configuration for this table
        table_config = self.boundary_config.get(table_name, {})
        boundary_columns = table_config.get('boundary_columns', [])
        core_sample_size = table_config.get('core_sample_size', self.default_sample_size)
        
        # If no boundary columns specified, fall back to random sampling
        if not boundary_columns or not all(col in df.columns for col in boundary_columns):
            self.logger.warning(f"No valid boundary columns specified for table {table_name}, falling back to random sampling")
            return RandomSampler(self.config_manager).sample(df, table_name, domain)
        
        # Initialize sampled data
        boundary_samples = []
        boundary_metadata = {}
        
        # Process each boundary column
        for column in boundary_columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[column]):
                self.logger.warning(f"Column {column} is not numeric, skipping boundary value sampling")
                continue
            
            # Get min and max values
            min_val = df[column].min()
            max_val = df[column].max()
            
            # Get rows with min and max values
            min_rows = df[df[column] == min_val]
            max_rows = df[df[column] == max_val]
            
            # Add to boundary samples
            boundary_samples.append(min_rows)
            boundary_samples.append(max_rows)
            
            # Store metadata
            boundary_metadata[column] = {
                'min_value': float(min_val),
                'max_value': float(max_val),
                'min_count': len(min_rows),
                'max_count': len(max_rows)
            }
        
        # Combine boundary samples
        boundary_df = pd.concat(boundary_samples).drop_duplicates() if boundary_samples else pd.DataFrame(columns=df.columns)
        
        # Sample core data (excluding boundary values)
        core_df = df[~df.index.isin(boundary_df.index)]
        
        # Calculate number of core rows to sample
        if isinstance(core_sample_size, float) and 0 < core_sample_size < 1:
            n_core_samples = int(len(core_df) * core_sample_size)
        else:
            n_core_samples = int(core_sample_size)
        
        # Ensure at least one row is sampled (if available)
        n_core_samples = max(1, min(n_core_samples, len(core_df)))
        
        # Sample core data
        sampled_core = core_df.sample(n=n_core_samples, random_state=42) if len(core_df) > 0 else pd.DataFrame(columns=df.columns)
        
        # Combine boundary and core samples
        sampled_df = pd.concat([boundary_df, sampled_core])
        
        # Create sampling result
        result = SamplingResult(
            table_name=table_name,
            domain=domain,
            original_row_count=len(df),
            sampled_row_count=len(sampled_df),
            sampling_method='boundary_value',
            sampling_rate=len(sampled_df) / len(df) if len(df) > 0 else 0.0,
            metadata={
                'boundary_columns': boundary_columns,
                'boundary_metadata': boundary_metadata,
                'boundary_count': len(boundary_df),
                'core_count': len(sampled_core),
                'core_sample_size': core_sample_size
            }
        )
        
        self.logger.info(f"Sampled {len(sampled_df)} rows from table {table_name} (reduction: {100 * (1 - result.sampling_rate):.2f}%)")
        return sampled_df, result


class AnomalyPreservingSampler(Sampler):
    """Sampler that preserves anomalies."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the anomaly-preserving sampler.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.base_sampler = None
    
    def initialize(self) -> None:
        """Initialize the anomaly-preserving sampler.
        
        Raises:
            ConfigurationError: If the sampler cannot be initialized.
        """
        # Create base sampler
        base_sampler_type = self.config_manager.get('core_sampling.methods.anomaly_preserving.base_sampler', 'random')
        
        if base_sampler_type == 'random':
            self.base_sampler = RandomSampler(self.config_manager)
        elif base_sampler_type == 'stratified':
            self.base_sampler = StratifiedSampler(self.config_manager)
        elif base_sampler_type == 'boundary_value':
            self.base_sampler = BoundaryValueSampler(self.config_manager)
        else:
            self.logger.warning(f"Unknown base sampler type: {base_sampler_type}, falling back to random")
            self.base_sampler = RandomSampler(self.config_manager)
        
        self.base_sampler.initialize()
        
        self.logger.info(f"Anomaly-preserving sampler initialized with base sampler: {base_sampler_type}")
    
    def sample(self, df: pd.DataFrame, table_name: str, domain: str, anomalies: pd.DataFrame = None) -> Tuple[pd.DataFrame, SamplingResult]:
        """Sample a DataFrame while preserving anomalies.
        
        Args:
            df: DataFrame to sample.
            table_name: Name of the table.
            domain: Domain of the table.
            anomalies: DataFrame containing anomalies to preserve.
        
        Returns:
            Tuple of (sampled DataFrame, sampling result).
        """
        self.logger.info(f"Sampling table {table_name} using anomaly-preserving sampling")
        
        # If no anomalies provided, fall back to base sampler
        if anomalies is None or anomalies.empty:
            self.logger.info(f"No anomalies provided for table {table_name}, falling back to base sampler")
            return self.base_sampler.sample(df, table_name, domain)
        
        # Remove anomalies from the DataFrame to sample
        normal_df = df[~df.index.isin(anomalies.index)]
        
        # Sample normal data using base sampler
        sampled_normal, base_result = self.base_sampler.sample(normal_df, table_name, domain)
        
        # Combine sampled normal data with anomalies
        sampled_df = pd.concat([sampled_normal, anomalies])
        
        # Create sampling result
        result = SamplingResult(
            table_name=table_name,
            domain=domain,
            original_row_count=len(df),
            sampled_row_count=len(sampled_df),
            sampling_method='anomaly_preserving',
            sampling_rate=len(sampled_df) / len(df) if len(df) > 0 else 0.0,
            metadata={
                'base_sampler': base_result.sampling_method,
                'normal_count': len(sampled_normal),
                'anomaly_count': len(anomalies),
                'base_metadata': base_result.metadata
            }
        )
        
        self.logger.info(f"Sampled {len(sampled_df)} rows from table {table_name} (reduction: {100 * (1 - result.sampling_rate):.2f}%)")
        return sampled_df, result


class SamplingPipeline(Pipeline):
    """Pipeline for data sampling."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the sampling pipeline.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.random_sampler = None
        self.stratified_sampler = None
        self.entity_based_sampler = None
        self.boundary_value_sampler = None
        self.anomaly_preserving_sampler = None
    
    def initialize(self) -> None:
        """Initialize the sampling pipeline.
        
        Raises:
            ConfigurationError: If the pipeline cannot be initialized.
        """
        super().initialize()
        
        # Initialize samplers
        self.random_sampler = RandomSampler(self.config_manager)
        self.random_sampler.initialize()
        
        self.stratified_sampler = StratifiedSampler(self.config_manager)
        self.stratified_sampler.initialize()
        
        self.entity_based_sampler = EntityBasedSampler(self.config_manager)
        self.entity_based_sampler.initialize()
        
        self.boundary_value_sampler = BoundaryValueSampler(self.config_manager)
        self.boundary_value_sampler.initialize()
        
        self.anomaly_preserving_sampler = AnomalyPreservingSampler(self.config_manager)
        self.anomaly_preserving_sampler.initialize()
        
        # Initialize steps
        self.steps = [
            PipelineStep(self.sample_data),
            PipelineStep(self.save_results)
        ]
        
        self.logger.info("Sampling pipeline initialized")
    
    def sample_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sample data using appropriate sampling methods.
        
        Args:
            data: Input data with normal data and anomalies.
        
        Returns:
            Updated data with sampled data.
        
        Raises:
            ProcessingError: If sampling fails.
        """
        self.logger.info("Sampling data")
        
        try:
            normal_data = data.get('normal_data', {})
            anomalies = data.get('anomalies', {})
            domains = data.get('domains', {})
            relationships = data.get('relationships', [])
            
            if not normal_data:
                self.logger.warning("No normal data found for sampling")
                return data
            
            # Convert relationships to dictionaries if needed
            relationship_dicts = []
            for r in relationships:
                if hasattr(r, 'to_dict'):
                    relationship_dicts.append(r.to_dict())
                else:
                    relationship_dicts.append(r)
            
            # Get method mapping
            method_mapping = self.config_manager.get('core_sampling.method_selection.method_mapping', {})
            default_method = method_mapping.get('default', 'random')
            
            # Store sampled data and results
            sampled_data = {}
            sampling_results = {}
            
            # Process entity-based sampling first
            entity_tables = set()
            
            if 'entity_based' in method_mapping.values():
                for entity_name in self.config_manager.get('core_sampling.methods.entity_based.entities', {}):
                    entity_results = self.entity_based_sampler.sample_entity(normal_data, entity_name, relationship_dicts)
                    
                    for table_name, (sampled_df, result) in entity_results.items():
                        sampled_data[table_name] = sampled_df
                        sampling_results[table_name] = result
                        entity_tables.add(table_name)
            
            # Process remaining tables
            for table_name, df in normal_data.items():
                # Skip tables already processed by entity-based sampling
                if table_name in entity_tables:
                    continue
                
                # Skip empty tables
                if df.empty:
                    self.logger.warning(f"Table {table_name} is empty, skipping")
                    sampled_data[table_name] = df
                    continue
                
                # Determine domain
                domain = 'unknown'
                for d, tables in domains.items():
                    if table_name in tables:
                        domain = d
                        break
                
                # Determine sampling method
                method = method_mapping.get(table_name, None)
                
                if method is None:
                    # Check for domain-specific method
                    method = method_mapping.get(f"domain:{domain}", default_method)
                
                # Get anomalies for this table
                table_anomalies = anomalies.get(table_name, pd.DataFrame())
                
                # Apply sampling method
                if method == 'random':
                    sampled_df, result = self.random_sampler.sample(df, table_name, domain)
                elif method == 'stratified':
                    sampled_df, result = self.stratified_sampler.sample(df, table_name, domain)
                elif method == 'boundary_value':
                    sampled_df, result = self.boundary_value_sampler.sample(df, table_name, domain)
                elif method == 'entity_based':
                    # Entity-based sampling should have been handled earlier
                    self.logger.warning(f"Table {table_name} mapped to entity_based sampling but not processed by entity sampler, falling back to random")
                    sampled_df, result = self.random_sampler.sample(df, table_name, domain)
                else:
                    self.logger.warning(f"Unknown sampling method: {method}, falling back to random")
                    sampled_df, result = self.random_sampler.sample(df, table_name, domain)
                
                # Preserve anomalies if present
                if not table_anomalies.empty:
                    sampled_df, result = self.anomaly_preserving_sampler.sample(
                        pd.concat([sampled_df, table_anomalies]),
                        table_name,
                        domain,
                        table_anomalies
                    )
                
                # Store results
                sampled_data[table_name] = sampled_df
                sampling_results[table_name] = result
            
            # Update data
            result = data.copy()
            result['sampled_data'] = sampled_data
            result['sampling_results'] = sampling_results
            
            # Log summary
            total_original = sum(len(df) for df in normal_data.values())
            total_sampled = sum(len(df) for df in sampled_data.values())
            
            self.logger.info(f"Sampled {total_sampled} rows from {total_original} rows (reduction: {100 * (1 - total_sampled / total_original):.2f}%)")
            
            return result
        except Exception as e:
            self.logger.error(f"Error sampling data: {str(e)}")
            raise ProcessingError(f"Error sampling data: {str(e)}")
    
    def save_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save sampling results.
        
        Args:
            data: Input data with sampled data and results.
        
        Returns:
            Updated data with saved results.
        
        Raises:
            ProcessingError: If saving results fails.
        """
        self.logger.info("Saving sampling results")
        
        try:
            sampled_data = data.get('sampled_data', {})
            sampling_results = data.get('sampling_results', {})
            
            if not sampled_data:
                self.logger.warning("No sampled data found to save")
                return data
            
            # Create output directories
            sampled_dir = os.path.join(self.output_dir, 'sampled_data')
            results_dir = os.path.join(self.output_dir, 'results')
            
            ensure_directory(sampled_dir)
            ensure_directory(results_dir)
            
            # Save sampled data
            sampled_files = {}
            for table_name, df in sampled_data.items():
                file_path = os.path.join(sampled_dir, f"{table_name}.csv")
                save_dataframe(df, file_path)
                sampled_files[table_name] = file_path
            
            # Save sampling results
            results_file = os.path.join(results_dir, 'sampling_results.json')
            save_json({table: result.to_dict() for table, result in sampling_results.items()}, results_file)
            
            # Save summary
            summary = {
                'table_counts': {table: len(df) for table, df in sampled_data.items()},
                'sampling_rates': {table: result.sampling_rate for table, result in sampling_results.items()},
                'sampling_methods': {table: result.sampling_method for table, result in sampling_results.items()},
                'total_original': sum(result.original_row_count for result in sampling_results.values()),
                'total_sampled': sum(result.sampled_row_count for result in sampling_results.values()),
                'overall_reduction': 1 - sum(result.sampled_row_count for result in sampling_results.values()) / sum(result.original_row_count for result in sampling_results.values()) if sum(result.original_row_count for result in sampling_results.values()) > 0 else 0
            }
            
            summary_file = os.path.join(results_dir, 'sampling_summary.json')
            save_json(summary, summary_file)
            
            # Update data
            result = data.copy()
            result['sampling_results_file'] = results_file
            result['sampling_summary_file'] = summary_file
            result['sampled_files'] = sampled_files
            
            self.logger.info("Sampling results saved")
            return result
        except Exception as e:
            self.logger.error(f"Error saving sampling results: {str(e)}")
            raise ProcessingError(f"Error saving sampling results: {str(e)}")

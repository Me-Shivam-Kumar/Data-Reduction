"""
Core Sampling Module for the Data Warehouse Subsampling Framework.

This module implements the third layer of the data subsampling architecture,
responsible for applying domain-specific sampling techniques to reduce data volume
while preserving data relationships and testing effectiveness.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import hashlib
import random

from ..common.base import Component, ConfigManager, PipelineStep, Pipeline, ProcessingError, ValidationError
from ..common.utils import (
    validate_data_frame, stratified_sampling, entity_based_subsetting, 
    boundary_value_extraction, preserve_referential_integrity
)

logger = logging.getLogger(__name__)


@dataclass
class SamplingResult:
    """Result of a sampling operation."""
    table_name: str
    domain: str
    original_row_count: int
    sampled_row_count: int
    reduction_ratio: float
    sampling_method: str
    preserved_anomalies: int
    total_anomalies: int
    anomaly_preservation_rate: float
    created_at: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the sampling result to a dictionary.
        
        Returns:
            Dictionary representation of the sampling result.
        """
        return {
            'table_name': self.table_name,
            'domain': self.domain,
            'original_row_count': self.original_row_count,
            'sampled_row_count': self.sampled_row_count,
            'reduction_ratio': self.reduction_ratio,
            'sampling_method': self.sampling_method,
            'preserved_anomalies': self.preserved_anomalies,
            'total_anomalies': self.total_anomalies,
            'anomaly_preservation_rate': self.anomaly_preservation_rate,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SamplingResult':
        """Create a SamplingResult from a dictionary.
        
        Args:
            data: Dictionary with sampling result information.
        
        Returns:
            SamplingResult instance.
        """
        created_at = None
        if data.get('created_at'):
            try:
                created_at = datetime.fromisoformat(data['created_at'])
            except (ValueError, TypeError):
                created_at = None
        
        return cls(
            table_name=data.get('table_name', ''),
            domain=data.get('domain', ''),
            original_row_count=data.get('original_row_count', 0),
            sampled_row_count=data.get('sampled_row_count', 0),
            reduction_ratio=data.get('reduction_ratio', 0.0),
            sampling_method=data.get('sampling_method', ''),
            preserved_anomalies=data.get('preserved_anomalies', 0),
            total_anomalies=data.get('total_anomalies', 0),
            anomaly_preservation_rate=data.get('anomaly_preservation_rate', 0.0),
            created_at=created_at
        )


class StratifiedSampler(Component):
    """Component for stratified sampling of data."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the stratified sampler.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.output_dir = None
        self.default_sample_size = None
        self.strata_config = None
    
    def initialize(self) -> None:
        """Initialize the stratified sampler.
        
        Raises:
            ConfigurationError: If the sampler cannot be initialized.
        """
        # Create output directory
        self.output_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'core_sampling',
            'stratified'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get configuration
        sampling_config = self.config_manager.get('core_sampling.methods.stratified', {})
        self.default_sample_size = sampling_config.get('default_sample_size', 0.1)
        self.strata_config = sampling_config.get('strata', {})
        
        self.logger.info(f"Stratified sampler initialized with default sample size {self.default_sample_size}")
    
    def validate(self) -> bool:
        """Validate the stratified sampler configuration and state.
        
        Returns:
            True if the sampler is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if not os.path.exists(self.output_dir):
            raise ValidationError(f"Output directory does not exist: {self.output_dir}")
        
        if self.default_sample_size <= 0 or self.default_sample_size > 1:
            raise ValidationError(f"Invalid default sample size: {self.default_sample_size}")
        
        return True
    
    def sample(self, df: pd.DataFrame, table_name: str, domain: str) -> Tuple[pd.DataFrame, SamplingResult]:
        """Perform stratified sampling on a DataFrame.
        
        Args:
            df: DataFrame to sample.
            table_name: Name of the table.
            domain: Domain name.
        
        Returns:
            Tuple of (sampled DataFrame, sampling result).
        """
        if not validate_data_frame(df):
            self.logger.warning(f"Invalid DataFrame for table {table_name}")
            return df, SamplingResult(
                table_name=table_name,
                domain=domain,
                original_row_count=0,
                sampled_row_count=0,
                reduction_ratio=1.0,
                sampling_method='stratified',
                preserved_anomalies=0,
                total_anomalies=0,
                anomaly_preservation_rate=0.0,
                created_at=datetime.now()
            )
        
        self.logger.info(f"Performing stratified sampling on table {table_name} with {len(df)} rows")
        
        # Get table-specific configuration
        table_config = self.strata_config.get(table_name, {})
        strata_columns = table_config.get('strata_columns', [])
        
        if not strata_columns:
            self.logger.warning(f"No strata columns specified for table {table_name}, using random sampling")
            
            # Perform random sampling
            sample_size = table_config.get('sample_size', self.default_sample_size)
            sampled_df = df.sample(frac=sample_size, random_state=42)
        else:
            # Get strata sample sizes
            strata_sample_sizes = table_config.get('strata_sample_sizes', {'default': self.default_sample_size})
            
            # Perform stratified sampling
            sampled_df = stratified_sampling(df, strata_columns, strata_sample_sizes)
        
        # Count anomalies
        total_anomalies = 0
        preserved_anomalies = 0
        
        if 'is_anomaly' in df.columns:
            total_anomalies = df['is_anomaly'].sum()
            if 'is_anomaly' in sampled_df.columns:
                preserved_anomalies = sampled_df['is_anomaly'].sum()
        
        # Calculate anomaly preservation rate
        anomaly_preservation_rate = 100.0
        if total_anomalies > 0:
            anomaly_preservation_rate = (preserved_anomalies / total_anomalies) * 100.0
        
        # Create sampling result
        result = SamplingResult(
            table_name=table_name,
            domain=domain,
            original_row_count=len(df),
            sampled_row_count=len(sampled_df),
            reduction_ratio=len(df) / len(sampled_df) if len(sampled_df) > 0 else 0.0,
            sampling_method='stratified',
            preserved_anomalies=preserved_anomalies,
            total_anomalies=total_anomalies,
            anomaly_preservation_rate=anomaly_preservation_rate,
            created_at=datetime.now()
        )
        
        # Save result
        self._save_result(result)
        
        self.logger.info(f"Stratified sampling reduced table {table_name} from {len(df)} to {len(sampled_df)} rows")
        return sampled_df, result
    
    def _save_result(self, result: SamplingResult) -> None:
        """Save sampling result to file.
        
        Args:
            result: SamplingResult instance.
        """
        # Convert to dictionary
        result_dict = result.to_dict()
        
        # Save to file
        output_file = os.path.join(self.output_dir, f"{result.table_name}_{result.domain}_result.json")
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Sampling result saved to {output_file}")


class EntityBasedSampler(Component):
    """Component for entity-based sampling of data."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the entity-based sampler.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.output_dir = None
        self.default_sample_size = None
        self.entity_config = None
    
    def initialize(self) -> None:
        """Initialize the entity-based sampler.
        
        Raises:
            ConfigurationError: If the sampler cannot be initialized.
        """
        # Create output directory
        self.output_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'core_sampling',
            'entity_based'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get configuration
        sampling_config = self.config_manager.get('core_sampling.methods.entity_based', {})
        self.default_sample_size = sampling_config.get('default_sample_size', 0.1)
        self.entity_config = sampling_config.get('entities', {})
        
        self.logger.info(f"Entity-based sampler initialized with default sample size {self.default_sample_size}")
    
    def validate(self) -> bool:
        """Validate the entity-based sampler configuration and state.
        
        Returns:
            True if the sampler is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if not os.path.exists(self.output_dir):
            raise ValidationError(f"Output directory does not exist: {self.output_dir}")
        
        if self.default_sample_size <= 0 or self.default_sample_size > 1:
            raise ValidationError(f"Invalid default sample size: {self.default_sample_size}")
        
        return True
    
    def sample(self, dfs: Dict[str, pd.DataFrame], domain: str) -> Tuple[Dict[str, pd.DataFrame], List[SamplingResult]]:
        """Perform entity-based sampling on a set of related DataFrames.
        
        Args:
            dfs: Dictionary mapping table names to DataFrames.
            domain: Domain name.
        
        Returns:
            Tuple of (dictionary of sampled DataFrames, list of sampling results).
        """
        if not dfs:
            self.logger.warning(f"No DataFrames to sample for domain {domain}")
            return {}, []
        
        self.logger.info(f"Performing entity-based sampling on {len(dfs)} tables in domain {domain}")
        
        # Get domain-specific configuration
        domain_config = self.entity_config.get(domain, {})
        
        # Get primary entity table
        primary_entity_table = domain_config.get('primary_entity_table')
        
        if not primary_entity_table or primary_entity_table not in dfs:
            self.logger.warning(f"Primary entity table not specified or not found for domain {domain}")
            return dfs, []
        
        # Get entity configuration
        entity_config = domain_config.get('entity_config', {})
        entity_config['sample_size'] = entity_config.get('sample_size', self.default_sample_size)
        
        # Get related tables
        related_tables = domain_config.get('related_tables', {})
        
        # Create dictionary of related DataFrames
        related_dfs = {}
        for table_name, df in dfs.items():
            if table_name != primary_entity_table:
                related_dfs[table_name] = df
        
        # Perform entity-based subsetting
        primary_df = dfs[primary_entity_table]
        sampled_primary_df, sampled_related_dfs = entity_based_subsetting(primary_df, entity_config, related_dfs)
        
        # Create result dictionary
        result_dfs = {primary_entity_table: sampled_primary_df}
        result_dfs.update(sampled_related_dfs)
        
        # Create sampling results
        sampling_results = []
        
        # Primary entity table result
        primary_result = self._create_sampling_result(primary_df, sampled_primary_df, primary_entity_table, domain, 'entity_based')
        sampling_results.append(primary_result)
        
        # Related tables results
        for table_name, sampled_df in sampled_related_dfs.items():
            original_df = dfs.get(table_name)
            if original_df is not None:
                result = self._create_sampling_result(original_df, sampled_df, table_name, domain, 'entity_based_related')
                sampling_results.append(result)
        
        # Save results
        for result in sampling_results:
            self._save_result(result)
        
        self.logger.info(f"Entity-based sampling completed for domain {domain}")
        return result_dfs, sampling_results
    
    def _create_sampling_result(self, original_df: pd.DataFrame, sampled_df: pd.DataFrame, 
                               table_name: str, domain: str, method: str) -> SamplingResult:
        """Create a sampling result for a table.
        
        Args:
            original_df: Original DataFrame.
            sampled_df: Sampled DataFrame.
            table_name: Name of the table.
            domain: Domain name.
            method: Sampling method.
        
        Returns:
            SamplingResult instance.
        """
        # Count anomalies
        total_anomalies = 0
        preserved_anomalies = 0
        
        if 'is_anomaly' in original_df.columns:
            total_anomalies = original_df['is_anomaly'].sum()
            if 'is_anomaly' in sampled_df.columns:
                preserved_anomalies = sampled_df['is_anomaly'].sum()
        
        # Calculate anomaly preservation rate
        anomaly_preservation_rate = 100.0
        if total_anomalies > 0:
            anomaly_preservation_rate = (preserved_anomalies / total_anomalies) * 100.0
        
        # Create sampling result
        return SamplingResult(
            table_name=table_name,
            domain=domain,
            original_row_count=len(original_df),
            sampled_row_count=len(sampled_df),
            reduction_ratio=len(original_df) / len(sampled_df) if len(sampled_df) > 0 else 0.0,
            sampling_method=method,
            preserved_anomalies=preserved_anomalies,
            total_anomalies=total_anomalies,
            anomaly_preservation_rate=anomaly_preservation_rate,
            created_at=datetime.now()
        )
    
    def _save_result(self, result: SamplingResult) -> None:
        """Save sampling result to file.
        
        Args:
            result: SamplingResult instance.
        """
        # Convert to dictionary
        result_dict = result.to_dict()
        
        # Save to file
        output_file = os.path.join(self.output_dir, f"{result.table_name}_{result.domain}_result.json")
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Sampling result saved to {output_file}")


class BoundaryValueSampler(Component):
    """Component for boundary value sampling of data."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the boundary value sampler.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.output_dir = None
        self.boundary_config = None
    
    def initialize(self) -> None:
        """Initialize the boundary value sampler.
        
        Raises:
            ConfigurationError: If the sampler cannot be initialized.
        """
        # Create output directory
        self.output_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'core_sampling',
            'boundary_value'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get configuration
        sampling_config = self.config_manager.get('core_sampling.methods.boundary_value', {})
        self.boundary_config = sampling_config.get('config', {})
        
        self.logger.info("Boundary value sampler initialized")
    
    def validate(self) -> bool:
        """Validate the boundary value sampler configuration and state.
        
        Returns:
            True if the sampler is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if not os.path.exists(self.output_dir):
            raise ValidationError(f"Output directory does not exist: {self.output_dir}")
        
        return True
    
    def sample(self, df: pd.DataFrame, table_name: str, domain: str) -> Tuple[pd.DataFrame, SamplingResult]:
        """Perform boundary value sampling on a DataFrame.
        
        Args:
            df: DataFrame to sample.
            table_name: Name of the table.
            domain: Domain name.
        
        Returns:
            Tuple of (sampled DataFrame, sampling result).
        """
        if not validate_data_frame(df):
            self.logger.warning(f"Invalid DataFrame for table {table_name}")
            return df, SamplingResult(
                table_name=table_name,
                domain=domain,
                original_row_count=0,
                sampled_row_count=0,
                reduction_ratio=1.0,
                sampling_method='boundary_value',
                preserved_anomalies=0,
                total_anomalies=0,
                anomaly_preservation_rate=0.0,
                created_at=datetime.now()
            )
        
        self.logger.info(f"Performing boundary value sampling on table {table_name} with {len(df)} rows")
        
        # Get table-specific configuration
        table_config = self.boundary_config.get(table_name, {})
        
        # Use default configuration if table-specific not found
        if not table_config:
            table_config = self.boundary_config.get('default', {
                'min_max': True,
                'percentiles': [5, 95],
                'outliers': True
            })
        
        # Perform boundary value extraction
        sampled_df = boundary_value_extraction(df, table_config)
        
        # Count anomalies
        total_anomalies = 0
        preserved_anomalies = 0
        
        if 'is_anomaly' in df.columns:
            total_anomalies = df['is_anomaly'].sum()
            if 'is_anomaly' in sampled_df.columns:
                preserved_anomalies = sampled_df['is_anomaly'].sum()
        
        # Calculate anomaly preservation rate
        anomaly_preservation_rate = 100.0
        if total_anomalies > 0:
            anomaly_preservation_rate = (preserved_anomalies / total_anomalies) * 100.0
        
        # Create sampling result
        result = SamplingResult(
            table_name=table_name,
            domain=domain,
            original_row_count=len(df),
            sampled_row_count=len(sampled_df),
            reduction_ratio=len(df) / len(sampled_df) if len(sampled_df) > 0 else 0.0,
            sampling_method='boundary_value',
            preserved_anomalies=preserved_anomalies,
            total_anomalies=total_anomalies,
            anomaly_preservation_rate=anomaly_preservation_rate,
            created_at=datetime.now()
        )
        
        # Save result
        self._save_result(result)
        
        self.logger.info(f"Boundary value sampling reduced table {table_name} from {len(df)} to {len(sampled_df)} rows")
        return sampled_df, result
    
    def _save_result(self, result: SamplingResult) -> None:
        """Save sampling result to file.
        
        Args:
            result: SamplingResult instance.
        """
        # Convert to dictionary
        result_dict = result.to_dict()
        
        # Save to file
        output_file = os.path.join(self.output_dir, f"{result.table_name}_{result.domain}_result.json")
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Sampling result saved to {output_file}")


class AnomalyPreservationSampler(Component):
    """Component for sampling data while preserving anomalies."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the anomaly preservation sampler.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.output_dir = None
        self.default_sample_size = None
        self.anomaly_config = None
    
    def initialize(self) -> None:
        """Initialize the anomaly preservation sampler.
        
        Raises:
            ConfigurationError: If the sampler cannot be initialized.
        """
        # Create output directory
        self.output_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'core_sampling',
            'anomaly_preservation'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get configuration
        sampling_config = self.config_manager.get('core_sampling.methods.anomaly_preservation', {})
        self.default_sample_size = sampling_config.get('default_sample_size', 0.1)
        self.anomaly_config = sampling_config.get('config', {})
        
        self.logger.info(f"Anomaly preservation sampler initialized with default sample size {self.default_sample_size}")
    
    def validate(self) -> bool:
        """Validate the anomaly preservation sampler configuration and state.
        
        Returns:
            True if the sampler is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if not os.path.exists(self.output_dir):
            raise ValidationError(f"Output directory does not exist: {self.output_dir}")
        
        if self.default_sample_size <= 0 or self.default_sample_size > 1:
            raise ValidationError(f"Invalid default sample size: {self.default_sample_size}")
        
        return True
    
    def sample(self, df: pd.DataFrame, table_name: str, domain: str) -> Tuple[pd.DataFrame, SamplingResult]:
        """Perform sampling while preserving anomalies.
        
        Args:
            df: DataFrame to sample.
            table_name: Name of the table.
            domain: Domain name.
        
        Returns:
            Tuple of (sampled DataFrame, sampling result).
        """
        if not validate_data_frame(df):
            self.logger.warning(f"Invalid DataFrame for table {table_name}")
            return df, SamplingResult(
                table_name=table_name,
                domain=domain,
                original_row_count=0,
                sampled_row_count=0,
                reduction_ratio=1.0,
                sampling_method='anomaly_preservation',
                preserved_anomalies=0,
                total_anomalies=0,
                anomaly_preservation_rate=0.0,
                created_at=datetime.now()
            )
        
        self.logger.info(f"Performing anomaly preservation sampling on table {table_name} with {len(df)} rows")
        
        # Check if anomaly column exists
        if 'is_anomaly' not in df.columns:
            self.logger.warning(f"No anomaly column found in table {table_name}, using random sampling")
            
            # Perform random sampling
            sample_size = self.anomaly_config.get('sample_size', self.default_sample_size)
            sampled_df = df.sample(frac=sample_size, random_state=42)
            
            # Create sampling result
            result = SamplingResult(
                table_name=table_name,
                domain=domain,
                original_row_count=len(df),
                sampled_row_count=len(sampled_df),
                reduction_ratio=len(df) / len(sampled_df) if len(sampled_df) > 0 else 0.0,
                sampling_method='random',
                preserved_anomalies=0,
                total_anomalies=0,
                anomaly_preservation_rate=0.0,
                created_at=datetime.now()
            )
            
            # Save result
            self._save_result(result)
            
            return sampled_df, result
        
        # Get table-specific configuration
        table_config = self.anomaly_config.get(table_name, {})
        sample_size = table_config.get('sample_size', self.default_sample_size)
        
        # Split DataFrame into anomalies and normal data
        anomalies_df = df[df['is_anomaly'] == True].copy()
        normal_df = df[df['is_anomaly'] == False].copy()
        
        # Sample normal data
        normal_sample_size = sample_size
        if len(anomalies_df) > 0:
            # Adjust normal sample size to account for anomalies
            target_size = int(len(df) * sample_size)
            normal_sample_size = max(0, (target_size - len(anomalies_df))) / len(normal_df)
        
        # Sample normal data
        sampled_normal_df = normal_df.sample(frac=normal_sample_size, random_state=42)
        
        # Combine anomalies and sampled normal data
        sampled_df = pd.concat([anomalies_df, sampled_normal_df])
        
        # Count anomalies
        total_anomalies = len(anomalies_df)
        preserved_anomalies = len(anomalies_df)
        
        # Calculate anomaly preservation rate
        anomaly_preservation_rate = 100.0
        
        # Create sampling result
        result = SamplingResult(
            table_name=table_name,
            domain=domain,
            original_row_count=len(df),
            sampled_row_count=len(sampled_df),
            reduction_ratio=len(df) / len(sampled_df) if len(sampled_df) > 0 else 0.0,
            sampling_method='anomaly_preservation',
            preserved_anomalies=preserved_anomalies,
            total_anomalies=total_anomalies,
            anomaly_preservation_rate=anomaly_preservation_rate,
            created_at=datetime.now()
        )
        
        # Save result
        self._save_result(result)
        
        self.logger.info(f"Anomaly preservation sampling reduced table {table_name} from {len(df)} to {len(sampled_df)} rows")
        self.logger.info(f"Preserved {preserved_anomalies} out of {total_anomalies} anomalies")
        
        return sampled_df, result
    
    def _save_result(self, result: SamplingResult) -> None:
        """Save sampling result to file.
        
        Args:
            result: SamplingResult instance.
        """
        # Convert to dictionary
        result_dict = result.to_dict()
        
        # Save to file
        output_file = os.path.join(self.output_dir, f"{result.table_name}_{result.domain}_result.json")
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Sampling result saved to {output_file}")


class SamplingMethodSelector(Component):
    """Component for selecting the appropriate sampling method for each table."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the sampling method selector.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.method_mapping = None
    
    def initialize(self) -> None:
        """Initialize the sampling method selector.
        
        Raises:
            ConfigurationError: If the selector cannot be initialized.
        """
        # Get configuration
        selector_config = self.config_manager.get('core_sampling.method_selection', {})
        self.method_mapping = selector_config.get('method_mapping', {})
        
        self.logger.info(f"Sampling method selector initialized with {len(self.method_mapping)} mappings")
    
    def validate(self) -> bool:
        """Validate the sampling method selector configuration and state.
        
        Returns:
            True if the selector is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        return True
    
    def select_method(self, table_name: str, domain: str) -> str:
        """Select the appropriate sampling method for a table.
        
        Args:
            table_name: Name of the table.
            domain: Domain name.
        
        Returns:
            Name of the selected sampling method.
        """
        # Check if there's a specific mapping for this table
        if table_name in self.method_mapping:
            return self.method_mapping[table_name]
        
        # Check if there's a domain-specific default
        domain_default = self.method_mapping.get(f"domain:{domain}")
        if domain_default:
            return domain_default
        
        # Use global default
        default_method = self.method_mapping.get('default', 'stratified')
        
        self.logger.info(f"Selected sampling method '{default_method}' for table {table_name} in domain {domain}")
        return default_method


class CoreSamplingPipeline(Pipeline):
    """Pipeline for core sampling."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the core sampling pipeline.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.stratified_sampler = None
        self.entity_based_sampler = None
        self.boundary_value_sampler = None
        self.anomaly_preservation_sampler = None
        self.method_selector = None
    
    def initialize(self) -> None:
        """Initialize the pipeline.
        
        Raises:
            ConfigurationError: If the pipeline cannot be initialized.
        """
        # Initialize components
        self.stratified_sampler = StratifiedSampler(self.config_manager)
        self.stratified_sampler.initialize()
        
        self.entity_based_sampler = EntityBasedSampler(self.config_manager)
        self.entity_based_sampler.initialize()
        
        self.boundary_value_sampler = BoundaryValueSampler(self.config_manager)
        self.boundary_value_sampler.initialize()
        
        self.anomaly_preservation_sampler = AnomalyPreservationSampler(self.config_manager)
        self.anomaly_preservation_sampler.initialize()
        
        self.method_selector = SamplingMethodSelector(self.config_manager)
        self.method_selector.initialize()
        
        # Add pipeline steps
        self.add_step(EntityBasedSamplingStep(self.config_manager, self.entity_based_sampler))
        self.add_step(TableSamplingStep(
            self.config_manager, 
            self.stratified_sampler,
            self.boundary_value_sampler,
            self.anomaly_preservation_sampler,
            self.method_selector
        ))
        self.add_step(ReferentialIntegrityStep(self.config_manager))
        
        self.logger.info("Core sampling pipeline initialized")
    
    def validate(self) -> bool:
        """Validate the pipeline configuration and state.
        
        Returns:
            True if the pipeline is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        # Validate components
        self.stratified_sampler.validate()
        self.entity_based_sampler.validate()
        self.boundary_value_sampler.validate()
        self.anomaly_preservation_sampler.validate()
        self.method_selector.validate()
        
        return True


class EntityBasedSamplingStep(PipelineStep):
    """Pipeline step for entity-based sampling."""
    
    def __init__(self, config_manager: ConfigManager, sampler: EntityBasedSampler):
        """Initialize the entity-based sampling step.
        
        Args:
            config_manager: Configuration manager instance.
            sampler: EntityBasedSampler instance.
        """
        super().__init__(config_manager)
        self.sampler = sampler
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the entity-based sampling step.
        
        Args:
            input_data: Dictionary with data, relationships, domain partitions, and anomalies.
        
        Returns:
            Dictionary with input data and sampled domain partitions.
        
        Raises:
            ProcessingError: If sampling fails.
        """
        self.logger.info("Performing entity-based sampling")
        
        if not input_data or 'domain_partitions' not in input_data:
            raise ProcessingError("No domain partitions to sample")
        
        # Check if entity-based sampling is enabled
        if not self.config_manager.get('core_sampling.methods.entity_based.enabled', True):
            self.logger.info("Entity-based sampling is disabled in configuration")
            return {**input_data, 'sampling_results': []}
        
        # Get domain partitions
        domain_partitions = input_data['domain_partitions']
        
        # Get entity-based domains
        entity_based_domains = self.config_manager.get('core_sampling.methods.entity_based.entities', {}).keys()
        
        # Sample each domain
        sampled_domain_partitions = {}
        all_sampling_results = []
        
        for domain, tables in domain_partitions.items():
            if domain in entity_based_domains:
                # Perform entity-based sampling
                try:
                    sampled_tables, sampling_results = self.sampler.sample(tables, domain)
                    sampled_domain_partitions[domain] = sampled_tables
                    all_sampling_results.extend(sampling_results)
                except Exception as e:
                    self.logger.error(f"Error performing entity-based sampling for domain {domain}: {str(e)}")
                    # Use original tables
                    sampled_domain_partitions[domain] = tables
            else:
                # Use original tables
                sampled_domain_partitions[domain] = tables
        
        self.logger.info(f"Entity-based sampling completed for {len(entity_based_domains)} domains")
        
        return {
            **input_data,
            'domain_partitions': sampled_domain_partitions,
            'sampling_results': all_sampling_results
        }


class TableSamplingStep(PipelineStep):
    """Pipeline step for table-level sampling."""
    
    def __init__(self, config_manager: ConfigManager, 
                stratified_sampler: StratifiedSampler,
                boundary_value_sampler: BoundaryValueSampler,
                anomaly_preservation_sampler: AnomalyPreservationSampler,
                method_selector: SamplingMethodSelector):
        """Initialize the table sampling step.
        
        Args:
            config_manager: Configuration manager instance.
            stratified_sampler: StratifiedSampler instance.
            boundary_value_sampler: BoundaryValueSampler instance.
            anomaly_preservation_sampler: AnomalyPreservationSampler instance.
            method_selector: SamplingMethodSelector instance.
        """
        super().__init__(config_manager)
        self.stratified_sampler = stratified_sampler
        self.boundary_value_sampler = boundary_value_sampler
        self.anomaly_preservation_sampler = anomaly_preservation_sampler
        self.method_selector = method_selector
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the table sampling step.
        
        Args:
            input_data: Dictionary with data, relationships, domain partitions, anomalies, and sampling results.
        
        Returns:
            Dictionary with input data and updated sampled domain partitions.
        
        Raises:
            ProcessingError: If sampling fails.
        """
        self.logger.info("Performing table-level sampling")
        
        if not input_data or 'domain_partitions' not in input_data:
            raise ProcessingError("No domain partitions to sample")
        
        # Get domain partitions
        domain_partitions = input_data['domain_partitions']
        
        # Get existing sampling results
        sampling_results = input_data.get('sampling_results', [])
        
        # Get entity-based domains (to skip tables that were already sampled)
        entity_based_domains = self.config_manager.get('core_sampling.methods.entity_based.entities', {}).keys()
        
        # Sample each table
        for domain, tables in domain_partitions.items():
            # Skip entity-based domains
            if domain in entity_based_domains:
                continue
            
            for table_name, df in tables.items():
                try:
                    # Select sampling method
                    method = self.method_selector.select_method(table_name, domain)
                    
                    # Apply sampling method
                    if method == 'stratified':
                        sampled_df, result = self.stratified_sampler.sample(df, table_name, domain)
                    elif method == 'boundary_value':
                        sampled_df, result = self.boundary_value_sampler.sample(df, table_name, domain)
                    elif method == 'anomaly_preservation':
                        sampled_df, result = self.anomaly_preservation_sampler.sample(df, table_name, domain)
                    else:
                        self.logger.warning(f"Unsupported sampling method '{method}' for table {table_name}, using stratified sampling")
                        sampled_df, result = self.stratified_sampler.sample(df, table_name, domain)
                    
                    # Update table in domain partition
                    domain_partitions[domain][table_name] = sampled_df
                    
                    # Add sampling result
                    sampling_results.append(result)
                except Exception as e:
                    self.logger.error(f"Error sampling table {table_name} in domain {domain}: {str(e)}")
                    # Continue with other tables
        
        self.logger.info("Table-level sampling completed")
        
        return {
            **input_data,
            'domain_partitions': domain_partitions,
            'sampling_results': sampling_results
        }


class ReferentialIntegrityStep(PipelineStep):
    """Pipeline step for preserving referential integrity."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the referential integrity step.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the referential integrity step.
        
        Args:
            input_data: Dictionary with data, relationships, domain partitions, anomalies, and sampling results.
        
        Returns:
            Dictionary with input data and updated sampled domain partitions.
        
        Raises:
            ProcessingError: If referential integrity preservation fails.
        """
        self.logger.info("Preserving referential integrity")
        
        if not input_data or 'domain_partitions' not in input_data:
            raise ProcessingError("No domain partitions to process")
        
        # Check if referential integrity preservation is enabled
        if not self.config_manager.get('core_sampling.referential_integrity.enabled', True):
            self.logger.info("Referential integrity preservation is disabled in configuration")
            return input_data
        
        # Get domain partitions
        domain_partitions = input_data['domain_partitions']
        
        # Get relationships
        relationships = input_data.get('relationships', [])
        
        if not relationships:
            self.logger.info("No relationships found, skipping referential integrity preservation")
            return input_data
        
        # Create a flat dictionary of all tables
        all_tables = {}
        for domain, tables in domain_partitions.items():
            for table_name, df in tables.items():
                all_tables[table_name] = df
        
        # Preserve referential integrity
        try:
            # Convert relationships to the format expected by preserve_referential_integrity
            relationship_dicts = []
            for rel in relationships:
                relationship_dicts.append({
                    'parent_table': rel.parent_table,
                    'parent_column': rel.parent_column,
                    'child_table': rel.child_table,
                    'child_column': rel.child_column
                })
            
            # Preserve referential integrity
            tables_with_integrity = preserve_referential_integrity(
                all_tables.get('main', pd.DataFrame()),  # Main table (if any)
                all_tables,  # All tables
                relationship_dicts  # Relationships
            )
            
            # Update domain partitions
            for domain, tables in domain_partitions.items():
                for table_name in tables.keys():
                    if table_name in tables_with_integrity:
                        domain_partitions[domain][table_name] = tables_with_integrity[table_name]
            
            self.logger.info("Referential integrity preserved")
        except Exception as e:
            self.logger.error(f"Error preserving referential integrity: {str(e)}")
            # Continue with pipeline
        
        return {
            **input_data,
            'domain_partitions': domain_partitions
        }

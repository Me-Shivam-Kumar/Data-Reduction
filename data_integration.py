"""
Data Integration module for the Data Warehouse Subsampling Framework.

This module provides components for integrating sampled data and anomalies
while maintaining referential integrity and creating coherent test datasets.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import json
from collections import defaultdict
import networkx as nx

from ..common.base import Component, ConfigManager, Pipeline, PipelineStep, IntegrationResult, ProcessingError
from ..common.utils import save_dataframe, load_dataframe, save_json, load_json, ensure_directory, create_sqlite_database

logger = logging.getLogger(__name__)


class ReferentialIntegrityIntegrator(Component):
    """Component for maintaining referential integrity in sampled data."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the referential integrity integrator.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.repair_strategy = None
    
    def initialize(self) -> None:
        """Initialize the referential integrity integrator.
        
        Raises:
            ConfigurationError: If the integrator cannot be initialized.
        """
        self.repair_strategy = self.config_manager.get('data_integration.referential_integrity.repair_strategy', 'add_missing')
        
        self.logger.info(f"Referential integrity integrator initialized with repair strategy: {self.repair_strategy}")
    
    def integrate(self, sampled_data: Dict[str, pd.DataFrame], relationships: List[Dict[str, Any]], original_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Integrate sampled data while maintaining referential integrity.
        
        Args:
            sampled_data: Dictionary mapping table names to sampled DataFrames.
            relationships: List of relationship dictionaries.
            original_data: Dictionary mapping table names to original DataFrames.
        
        Returns:
            Dictionary mapping table names to integrated DataFrames.
        """
        self.logger.info("Integrating sampled data with referential integrity")
        
        # Create a copy of sampled data
        integrated_data = {table: df.copy() for table, df in sampled_data.items()}
        
        # Create a graph of table relationships
        G = nx.DiGraph()
        
        # Add nodes for each table
        for table_name in sampled_data.keys():
            G.add_node(table_name)
        
        # Add edges for relationships
        for relationship in relationships:
            parent_table = relationship.get('parent_table')
            child_table = relationship.get('child_table')
            
            if parent_table in sampled_data and child_table in sampled_data:
                G.add_edge(parent_table, child_table, relationship=relationship)
        
        # Process tables in topological order (parents before children)
        try:
            for table_name in nx.topological_sort(G):
                # Get incoming edges (relationships where this table is the child)
                for parent_table in G.predecessors(table_name):
                    edge_data = G.get_edge_data(parent_table, table_name)
                    relationship = edge_data.get('relationship')
                    
                    # Repair referential integrity
                    integrated_data = self._repair_integrity(
                        integrated_data,
                        parent_table,
                        relationship.get('parent_column'),
                        table_name,
                        relationship.get('child_column'),
                        original_data
                    )
        except nx.NetworkXUnfeasible:
            self.logger.warning("Cyclic relationships detected, processing tables in arbitrary order")
            
            # Process all relationships
            for relationship in relationships:
                parent_table = relationship.get('parent_table')
                parent_column = relationship.get('parent_column')
                child_table = relationship.get('child_table')
                child_column = relationship.get('child_column')
                
                if parent_table in integrated_data and child_table in integrated_data:
                    integrated_data = self._repair_integrity(
                        integrated_data,
                        parent_table,
                        parent_column,
                        child_table,
                        child_column,
                        original_data
                    )
        
        return integrated_data
    
    def _repair_integrity(self, data: Dict[str, pd.DataFrame], parent_table: str, parent_column: str, child_table: str, child_column: str, original_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Repair referential integrity for a specific relationship.
        
        Args:
            data: Dictionary mapping table names to DataFrames.
            parent_table: Name of the parent table.
            parent_column: Name of the parent column.
            child_table: Name of the child table.
            child_column: Name of the child column.
            original_data: Dictionary mapping table names to original DataFrames.
        
        Returns:
            Updated data dictionary.
        """
        self.logger.info(f"Repairing integrity: {child_table}.{child_column} -> {parent_table}.{parent_column}")
        
        # Get parent and child DataFrames
        parent_df = data.get(parent_table)
        child_df = data.get(child_table)
        
        if parent_df is None or child_df is None:
            self.logger.warning(f"Missing table: {parent_table if parent_df is None else child_table}")
            return data
        
        # Check if columns exist
        if parent_column not in parent_df.columns:
            self.logger.warning(f"Parent column {parent_column} not found in table {parent_table}")
            return data
        
        if child_column not in child_df.columns:
            self.logger.warning(f"Child column {child_column} not found in table {child_table}")
            return data
        
        # Get parent values
        parent_values = set(parent_df[parent_column].dropna())
        
        # Get child values
        child_values = set(child_df[child_column].dropna())
        
        # Find orphaned values (in child but not in parent)
        orphaned_values = child_values - parent_values
        
        if not orphaned_values:
            self.logger.info(f"No orphaned values found for {child_table}.{child_column}")
            return data
        
        self.logger.info(f"Found {len(orphaned_values)} orphaned values for {child_table}.{child_column}")
        
        # Apply repair strategy
        if self.repair_strategy == 'add_missing':
            # Add missing parent records
            if parent_table in original_data:
                original_parent_df = original_data[parent_table]
                
                if parent_column in original_parent_df.columns:
                    # Find matching records in original data
                    missing_parents = original_parent_df[original_parent_df[parent_column].isin(orphaned_values)]
                    
                    if not missing_parents.empty:
                        # Add missing parents to integrated data
                        data[parent_table] = pd.concat([parent_df, missing_parents]).drop_duplicates()
                        
                        self.logger.info(f"Added {len(missing_parents)} missing parent records to {parent_table}")
                    else:
                        self.logger.warning(f"No matching parent records found in original data for {parent_table}")
                else:
                    self.logger.warning(f"Parent column {parent_column} not found in original table {parent_table}")
            else:
                self.logger.warning(f"Original data not available for table {parent_table}")
        
        elif self.repair_strategy == 'remove_orphans':
            # Remove orphaned child records
            orphan_mask = child_df[child_column].isin(orphaned_values)
            data[child_table] = child_df[~orphan_mask]
            
            self.logger.info(f"Removed {orphan_mask.sum()} orphaned records from {child_table}")
        
        elif self.repair_strategy == 'nullify_orphans':
            # Set orphaned foreign keys to NULL
            orphan_mask = child_df[child_column].isin(orphaned_values)
            child_df.loc[orphan_mask, child_column] = None
            data[child_table] = child_df
            
            self.logger.info(f"Nullified {orphan_mask.sum()} orphaned foreign keys in {child_table}")
        
        else:
            self.logger.warning(f"Unknown repair strategy: {self.repair_strategy}")
        
        return data


class AnomalyContextIntegrator(Component):
    """Component for integrating anomalies with their context."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the anomaly context integrator.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
    
    def initialize(self) -> None:
        """Initialize the anomaly context integrator.
        
        Raises:
            ConfigurationError: If the integrator cannot be initialized.
        """
        self.logger.info("Anomaly context integrator initialized")
    
    def integrate(self, integrated_data: Dict[str, pd.DataFrame], anomalies: Dict[str, pd.DataFrame], anomaly_context: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """Integrate anomalies and their context into the data.
        
        Args:
            integrated_data: Dictionary mapping table names to integrated DataFrames.
            anomalies: Dictionary mapping table names to anomaly DataFrames.
            anomaly_context: Dictionary mapping table names to dictionaries mapping related table names to context DataFrames.
        
        Returns:
            Dictionary mapping table names to DataFrames with integrated anomalies and context.
        """
        self.logger.info("Integrating anomalies and context")
        
        # Create a copy of integrated data
        result = {table: df.copy() for table, df in integrated_data.items()}
        
        # Add anomalies to their respective tables
        for table_name, anomalies_df in anomalies.items():
            if table_name in result:
                # Combine with existing data, removing duplicates
                result[table_name] = pd.concat([result[table_name], anomalies_df]).drop_duplicates()
                
                self.logger.info(f"Added {len(anomalies_df)} anomalies to table {table_name}")
            else:
                # Add as a new table
                result[table_name] = anomalies_df
                
                self.logger.info(f"Added new table {table_name} with {len(anomalies_df)} anomalies")
        
        # Add context to their respective tables
        for table_name, context_tables in anomaly_context.items():
            for context_table, context_df in context_tables.items():
                if context_table in result:
                    # Combine with existing data, removing duplicates
                    result[context_table] = pd.concat([result[context_table], context_df]).drop_duplicates()
                    
                    self.logger.info(f"Added {len(context_df)} context records to table {context_table} for anomalies in {table_name}")
                else:
                    # Add as a new table
                    result[context_table] = context_df
                    
                    self.logger.info(f"Added new table {context_table} with {len(context_df)} context records for anomalies in {table_name}")
        
        return result


class DatasetCreator(Component):
    """Component for creating purpose-specific datasets."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the dataset creator.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.datasets_config = None
    
    def initialize(self) -> None:
        """Initialize the dataset creator.
        
        Raises:
            ConfigurationError: If the creator cannot be initialized.
        """
        self.datasets_config = self.config_manager.get('data_integration.datasets', [])
        
        self.logger.info(f"Dataset creator initialized with {len(self.datasets_config)} dataset configurations")
    
    def create_datasets(self, integrated_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Create purpose-specific datasets.
        
        Args:
            integrated_data: Dictionary mapping table names to integrated DataFrames.
        
        Returns:
            Dictionary mapping dataset names to dictionaries mapping table names to DataFrames.
        """
        self.logger.info("Creating purpose-specific datasets")
        
        datasets = {}
        
        # Create default dataset with all tables
        datasets['default'] = {table: df.copy() for table, df in integrated_data.items()}
        
        # Create configured datasets
        for dataset_config in self.datasets_config:
            dataset_name = dataset_config.get('name')
            
            if not dataset_name:
                self.logger.warning("Dataset configuration missing name, skipping")
                continue
            
            self.logger.info(f"Creating dataset: {dataset_name}")
            
            # Get included and excluded tables
            include_tables = dataset_config.get('include_tables', [])
            exclude_tables = dataset_config.get('exclude_tables', [])
            
            # Get included and excluded domains
            include_domains = dataset_config.get('include_domains', [])
            exclude_domains = dataset_config.get('exclude_domains', [])
            
            # Get domain mappings
            domain_mappings = dataset_config.get('domain_mappings', {})
            
            # Create dataset
            dataset = {}
            
            # Add tables based on inclusion/exclusion lists
            for table_name, df in integrated_data.items():
                # Check if table is explicitly included or excluded
                if include_tables and table_name not in include_tables:
                    continue
                
                if table_name in exclude_tables:
                    continue
                
                # Check if table's domain is included or excluded
                table_domain = None
                for domain, tables in domain_mappings.items():
                    if table_name in tables:
                        table_domain = domain
                        break
                
                if include_domains and table_domain not in include_domains:
                    continue
                
                if table_domain in exclude_domains:
                    continue
                
                # Add table to dataset
                dataset[table_name] = df.copy()
            
            # Add dataset if not empty
            if dataset:
                datasets[dataset_name] = dataset
                self.logger.info(f"Created dataset {dataset_name} with {len(dataset)} tables")
            else:
                self.logger.warning(f"Dataset {dataset_name} is empty, skipping")
        
        return datasets


class DataIntegrationPipeline(Pipeline):
    """Pipeline for data integration."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the data integration pipeline.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.referential_integrity_integrator = None
        self.anomaly_context_integrator = None
        self.dataset_creator = None
    
    def initialize(self) -> None:
        """Initialize the data integration pipeline.
        
        Raises:
            ConfigurationError: If the pipeline cannot be initialized.
        """
        super().initialize()
        
        # Initialize components
        self.referential_integrity_integrator = ReferentialIntegrityIntegrator(self.config_manager)
        self.referential_integrity_integrator.initialize()
        
        self.anomaly_context_integrator = AnomalyContextIntegrator(self.config_manager)
        self.anomaly_context_integrator.initialize()
        
        self.dataset_creator = DatasetCreator(self.config_manager)
        self.dataset_creator.initialize()
        
        # Initialize steps
        self.steps = [
            PipelineStep(self.integrate_referential_integrity),
            PipelineStep(self.integrate_anomalies),
            PipelineStep(self.create_datasets),
            PipelineStep(self.save_results)
        ]
        
        self.logger.info("Data integration pipeline initialized")
    
    def integrate_referential_integrity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate sampled data with referential integrity.
        
        Args:
            data: Input data with sampled data and relationships.
        
        Returns:
            Updated data with integrated data.
        
        Raises:
            ProcessingError: If integration fails.
        """
        self.logger.info("Integrating referential integrity")
        
        try:
            sampled_data = data.get('sampled_data', {})
            relationships = data.get('relationships', [])
            original_data = data.get('original_data', {})
            
            if not sampled_data:
                self.logger.warning("No sampled data found for integration")
                return data
            
            # Convert relationships to dictionaries if needed
            relationship_dicts = []
            for r in relationships:
                if hasattr(r, 'to_dict'):
                    relationship_dicts.append(r.to_dict())
                else:
                    relationship_dicts.append(r)
            
            # Integrate referential integrity
            integrated_data = self.referential_integrity_integrator.integrate(sampled_data, relationship_dicts, original_data)
            
            # Update data
            result = data.copy()
            result['integrated_data'] = integrated_data
            
            # Log summary
            total_sampled = sum(len(df) for df in sampled_data.values())
            total_integrated = sum(len(df) for df in integrated_data.values())
            
            self.logger.info(f"Integrated data: {total_integrated} rows (from {total_sampled} sampled rows)")
            
            return result
        except Exception as e:
            self.logger.error(f"Error integrating referential integrity: {str(e)}")
            raise ProcessingError(f"Error integrating referential integrity: {str(e)}")
    
    def integrate_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate anomalies and their context.
        
        Args:
            data: Input data with integrated data, anomalies, and anomaly context.
        
        Returns:
            Updated data with anomalies integrated.
        
        Raises:
            ProcessingError: If integration fails.
        """
        self.logger.info("Integrating anomalies and context")
        
        try:
            integrated_data = data.get('integrated_data', {})
            anomalies = data.get('anomalies', {})
            anomaly_context = data.get('anomaly_context', {})
            
            if not integrated_data:
                self.logger.warning("No integrated data found for anomaly integration")
                return data
            
            # Integrate anomalies and context
            anomaly_integrated_data = self.anomaly_context_integrator.integrate(integrated_data, anomalies, anomaly_context)
            
            # Update data
            result = data.copy()
            result['anomaly_integrated_data'] = anomaly_integrated_data
            
            # Log summary
            total_integrated = sum(len(df) for df in integrated_data.values())
            total_anomaly_integrated = sum(len(df) for df in anomaly_integrated_data.values())
            
            self.logger.info(f"Integrated anomalies: {total_anomaly_integrated} rows (from {total_integrated} integrated rows)")
            
            return result
        except Exception as e:
            self.logger.error(f"Error integrating anomalies: {str(e)}")
            raise ProcessingError(f"Error integrating anomalies: {str(e)}")
    
    def create_datasets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create purpose-specific datasets.
        
        Args:
            data: Input data with anomaly-integrated data.
        
        Returns:
            Updated data with datasets.
        
        Raises:
            ProcessingError: If dataset creation fails.
        """
        self.logger.info("Creating datasets")
        
        try:
            anomaly_integrated_data = data.get('anomaly_integrated_data', {})
            
            if not anomaly_integrated_data:
                self.logger.warning("No anomaly-integrated data found for dataset creation")
                return data
            
            # Create datasets
            datasets = self.dataset_creator.create_datasets(anomaly_integrated_data)
            
            # Update data
            result = data.copy()
            result['datasets'] = datasets
            
            # Log summary
            self.logger.info(f"Created {len(datasets)} datasets")
            
            return result
        except Exception as e:
            self.logger.error(f"Error creating datasets: {str(e)}")
            raise ProcessingError(f"Error creating datasets: {str(e)}")
    
    def save_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save integration results.
        
        Args:
            data: Input data with datasets.
        
        Returns:
            Updated data with saved results.
        
        Raises:
            ProcessingError: If saving results fails.
        """
        self.logger.info("Saving integration results")
        
        try:
            datasets = data.get('datasets', {})
            
            if not datasets:
                self.logger.warning("No datasets found to save")
                return data
            
            # Create output directories
            datasets_dir = os.path.join(self.output_dir, 'datasets')
            ensure_directory(datasets_dir)
            
            # Save datasets
            dataset_files = {}
            dataset_results = {}
            
            for dataset_name, dataset_tables in datasets.items():
                # Create dataset directory
                dataset_dir = os.path.join(datasets_dir, dataset_name)
                ensure_directory(dataset_dir)
                
                # Save tables
                table_files = {}
                
                for table_name, df in dataset_tables.items():
                    file_path = os.path.join(dataset_dir, f"{table_name}.csv")
                    save_dataframe(df, file_path)
                    table_files[table_name] = file_path
                
                # Create SQLite database
                db_path = os.path.join(dataset_dir, f"{dataset_name}.db")
                create_sqlite_database(db_path, dataset_tables)
                
                # Create integration result
                result = IntegrationResult(
                    name=dataset_name,
                    tables=list(dataset_tables.keys()),
                    row_counts={table: len(df) for table, df in dataset_tables.items()},
                    metadata={
                        'table_files': table_files,
                        'database_file': db_path
                    }
                )
                
                # Store result
                dataset_results[dataset_name] = result
                dataset_files[dataset_name] = {
                    'tables': table_files,
                    'database': db_path
                }
            
            # Save results
            results_file = os.path.join(self.output_dir, 'integration_results.json')
            save_json({name: result.to_dict() for name, result in dataset_results.items()}, results_file)
            
            # Save summary
            summary = {
                'dataset_counts': {name: len(tables) for name, tables in datasets.items()},
                'dataset_row_counts': {name: sum(len(df) for df in tables.values()) for name, tables in datasets.items()},
                'dataset_files': dataset_files
            }
            
            summary_file = os.path.join(self.output_dir, 'integration_summary.json')
            save_json(summary, summary_file)
            
            # Update data
            result = data.copy()
            result['integration_results'] = {
                'dataset_results': dataset_results,
                'results_file': results_file,
                'summary_file': summary_file,
                'dataset_files': dataset_files
            }
            
            self.logger.info("Integration results saved")
            return result
        except Exception as e:
            self.logger.error(f"Error saving integration results: {str(e)}")
            raise ProcessingError(f"Error saving integration results: {str(e)}")

"""
Data Integration Module for the Data Warehouse Subsampling Framework.

This module implements the fourth layer of the data subsampling architecture,
responsible for maintaining referential integrity across sampled datasets,
ensuring anomaly-to-normal data relationships are preserved, and creating
purpose-specific integrated test datasets.
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
import networkx as nx

from ..common.base import Component, ConfigManager, PipelineStep, Pipeline, ProcessingError, ValidationError
from ..common.utils import validate_data_frame, preserve_referential_integrity

logger = logging.getLogger(__name__)


@dataclass
class IntegrationResult:
    """Result of a data integration operation."""
    name: str
    tables: List[str]
    original_row_count: int
    integrated_row_count: int
    anomaly_count: int
    relationships_preserved: int
    total_relationships: int
    relationship_preservation_rate: float
    created_at: datetime = None
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the integration result to a dictionary.
        
        Returns:
            Dictionary representation of the integration result.
        """
        return {
            'name': self.name,
            'tables': self.tables,
            'original_row_count': self.original_row_count,
            'integrated_row_count': self.integrated_row_count,
            'anomaly_count': self.anomaly_count,
            'relationships_preserved': self.relationships_preserved,
            'total_relationships': self.total_relationships,
            'relationship_preservation_rate': self.relationship_preservation_rate,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntegrationResult':
        """Create an IntegrationResult from a dictionary.
        
        Args:
            data: Dictionary with integration result information.
        
        Returns:
            IntegrationResult instance.
        """
        created_at = None
        if data.get('created_at'):
            try:
                created_at = datetime.fromisoformat(data['created_at'])
            except (ValueError, TypeError):
                created_at = None
        
        return cls(
            name=data.get('name', ''),
            tables=data.get('tables', []),
            original_row_count=data.get('original_row_count', 0),
            integrated_row_count=data.get('integrated_row_count', 0),
            anomaly_count=data.get('anomaly_count', 0),
            relationships_preserved=data.get('relationships_preserved', 0),
            total_relationships=data.get('total_relationships', 0),
            relationship_preservation_rate=data.get('relationship_preservation_rate', 0.0),
            created_at=created_at
        )


class ReferentialIntegrityIntegrator(Component):
    """Component for integrating data with referential integrity."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the referential integrity integrator.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.output_dir = None
    
    def initialize(self) -> None:
        """Initialize the referential integrity integrator.
        
        Raises:
            ConfigurationError: If the integrator cannot be initialized.
        """
        # Create output directory
        self.output_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'data_integration',
            'referential_integrity'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info("Referential integrity integrator initialized")
    
    def validate(self) -> bool:
        """Validate the referential integrity integrator configuration and state.
        
        Returns:
            True if the integrator is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if not os.path.exists(self.output_dir):
            raise ValidationError(f"Output directory does not exist: {self.output_dir}")
        
        return True
    
    def integrate(self, domain_partitions: Dict[str, Dict[str, pd.DataFrame]], 
                 relationships: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], IntegrationResult]:
        """Integrate data with referential integrity.
        
        Args:
            domain_partitions: Dictionary mapping domain names to dictionaries of table DataFrames.
            relationships: List of relationship dictionaries.
        
        Returns:
            Tuple of (integrated domain partitions, integration result).
        """
        self.logger.info(f"Integrating data with referential integrity for {len(domain_partitions)} domains")
        
        # Create a flat dictionary of all tables
        all_tables = {}
        for domain, tables in domain_partitions.items():
            for table_name, df in tables.items():
                all_tables[table_name] = df
        
        # Count original rows
        original_row_count = sum(len(df) for df in all_tables.values())
        
        # Count anomalies
        anomaly_count = 0
        for df in all_tables.values():
            if 'is_anomaly' in df.columns:
                anomaly_count += df['is_anomaly'].sum()
        
        # Preserve referential integrity
        try:
            # Preserve referential integrity
            main_table = next(iter(all_tables.values())) if all_tables else pd.DataFrame()
            tables_with_integrity = preserve_referential_integrity(main_table, all_tables, relationships)
            
            # Update domain partitions
            integrated_domain_partitions = {}
            for domain, tables in domain_partitions.items():
                integrated_domain_partitions[domain] = {}
                for table_name, df in tables.items():
                    if table_name in tables_with_integrity:
                        integrated_domain_partitions[domain][table_name] = tables_with_integrity[table_name]
                    else:
                        integrated_domain_partitions[domain][table_name] = df
            
            # Count integrated rows
            integrated_row_count = sum(len(df) for df in tables_with_integrity.values())
            
            # Count preserved relationships
            relationships_preserved = self._count_preserved_relationships(tables_with_integrity, relationships)
            total_relationships = len(relationships)
            
            # Calculate relationship preservation rate
            relationship_preservation_rate = 100.0
            if total_relationships > 0:
                relationship_preservation_rate = (relationships_preserved / total_relationships) * 100.0
            
            # Create integration result
            result = IntegrationResult(
                name="referential_integrity",
                tables=list(all_tables.keys()),
                original_row_count=original_row_count,
                integrated_row_count=integrated_row_count,
                anomaly_count=anomaly_count,
                relationships_preserved=relationships_preserved,
                total_relationships=total_relationships,
                relationship_preservation_rate=relationship_preservation_rate,
                created_at=datetime.now()
            )
            
            # Save result
            self._save_result(result)
            
            self.logger.info(f"Referential integrity integration completed with {relationships_preserved}/{total_relationships} relationships preserved")
            return integrated_domain_partitions, result
        except Exception as e:
            self.logger.error(f"Error integrating data with referential integrity: {str(e)}")
            # Return original domain partitions
            return domain_partitions, IntegrationResult(
                name="referential_integrity",
                tables=list(all_tables.keys()),
                original_row_count=original_row_count,
                integrated_row_count=original_row_count,
                anomaly_count=anomaly_count,
                relationships_preserved=0,
                total_relationships=len(relationships),
                relationship_preservation_rate=0.0,
                created_at=datetime.now()
            )
    
    def _count_preserved_relationships(self, tables: Dict[str, pd.DataFrame], 
                                     relationships: List[Dict[str, Any]]) -> int:
        """Count the number of preserved relationships.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
            relationships: List of relationship dictionaries.
        
        Returns:
            Number of preserved relationships.
        """
        preserved_count = 0
        
        for rel in relationships:
            parent_table = rel.get('parent_table')
            parent_column = rel.get('parent_column')
            child_table = rel.get('child_table')
            child_column = rel.get('child_column')
            
            # Skip if any table is missing
            if parent_table not in tables or child_table not in tables:
                continue
            
            # Get the DataFrames
            parent_df = tables[parent_table]
            child_df = tables[child_table]
            
            # Skip if any column is missing
            if parent_column not in parent_df.columns or child_column not in child_df.columns:
                continue
            
            # Get the values in the child DataFrame
            child_values = set(child_df[child_column].dropna().unique())
            
            # Get the values in the parent DataFrame
            parent_values = set(parent_df[parent_column].dropna().unique())
            
            # Check if all child values are in the parent
            if child_values.issubset(parent_values):
                preserved_count += 1
        
        return preserved_count
    
    def _save_result(self, result: IntegrationResult) -> None:
        """Save integration result to file.
        
        Args:
            result: IntegrationResult instance.
        """
        # Convert to dictionary
        result_dict = result.to_dict()
        
        # Save to file
        output_file = os.path.join(self.output_dir, f"{result.name}_result.json")
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Integration result saved to {output_file}")


class AnomalyContextIntegrator(Component):
    """Component for integrating anomalies with their context."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the anomaly context integrator.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.output_dir = None
        self.context_config = None
    
    def initialize(self) -> None:
        """Initialize the anomaly context integrator.
        
        Raises:
            ConfigurationError: If the integrator cannot be initialized.
        """
        # Create output directory
        self.output_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'data_integration',
            'anomaly_context'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get configuration
        self.context_config = self.config_manager.get('data_integration.anomaly_context', {})
        
        self.logger.info("Anomaly context integrator initialized")
    
    def validate(self) -> bool:
        """Validate the anomaly context integrator configuration and state.
        
        Returns:
            True if the integrator is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if not os.path.exists(self.output_dir):
            raise ValidationError(f"Output directory does not exist: {self.output_dir}")
        
        return True
    
    def integrate(self, domain_partitions: Dict[str, Dict[str, pd.DataFrame]], 
                 anomalies: List[Dict[str, Any]],
                 relationships: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], IntegrationResult]:
        """Integrate anomalies with their context.
        
        Args:
            domain_partitions: Dictionary mapping domain names to dictionaries of table DataFrames.
            anomalies: List of anomaly dictionaries.
            relationships: List of relationship dictionaries.
        
        Returns:
            Tuple of (integrated domain partitions, integration result).
        """
        self.logger.info(f"Integrating {len(anomalies)} anomalies with their context")
        
        # Create a flat dictionary of all tables
        all_tables = {}
        for domain, tables in domain_partitions.items():
            for table_name, df in tables.items():
                all_tables[table_name] = df
        
        # Count original rows
        original_row_count = sum(len(df) for df in all_tables.values())
        
        # Count anomalies
        anomaly_count = len(anomalies)
        
        # Create a graph of table relationships
        graph = nx.DiGraph()
        
        # Add nodes for each table
        for table_name in all_tables.keys():
            graph.add_node(table_name)
        
        # Add edges for relationships
        for rel in relationships:
            parent_table = rel.get('parent_table')
            child_table = rel.get('child_table')
            
            if parent_table in all_tables and child_table in all_tables:
                graph.add_edge(parent_table, child_table, relationship=rel)
                graph.add_edge(child_table, parent_table, relationship=rel)
        
        # Process each anomaly
        context_rows_added = 0
        relationships_preserved = 0
        
        for anomaly in anomalies:
            table_name = anomaly.get('table_name')
            original_row = anomaly.get('original_row', {})
            
            if table_name not in all_tables:
                continue
            
            # Get context depth
            context_depth = self.context_config.get('context_depth', 1)
            
            # Find related tables within context depth
            related_tables = set()
            for target_table in all_tables.keys():
                if target_table == table_name:
                    continue
                
                try:
                    # Find shortest path
                    path = nx.shortest_path(graph, table_name, target_table)
                    
                    # Check if within context depth
                    if len(path) - 1 <= context_depth:
                        related_tables.add(target_table)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # No path found
                    pass
            
            # Add context rows for each related table
            for related_table in related_tables:
                # Find the path to the related table
                try:
                    path = nx.shortest_path(graph, table_name, related_table)
                    
                    # Follow the path to find related rows
                    current_table = table_name
                    current_values = {k: v for k, v in original_row.items()}
                    
                    for next_table in path[1:]:
                        # Find the relationship between current_table and next_table
                        for edge_data in graph.get_edge_data(current_table, next_table).values():
                            rel = edge_data.get('relationship')
                            if rel:
                                # Determine the direction
                                if rel.get('parent_table') == current_table and rel.get('child_table') == next_table:
                                    # current_table is parent, next_table is child
                                    parent_column = rel.get('parent_column')
                                    child_column = rel.get('child_column')
                                    
                                    if parent_column in current_values:
                                        # Find rows in next_table where child_column matches parent_column value
                                        value = current_values[parent_column]
                                        related_rows = all_tables[next_table][all_tables[next_table][child_column] == value]
                                        
                                        if not related_rows.empty:
                                            # Add related rows to the table
                                            current_table = next_table
                                            current_values = related_rows.iloc[0].to_dict()
                                            
                                            # Add rows to the table if not already present
                                            for _, row in related_rows.iterrows():
                                                row_dict = row.to_dict()
                                                
                                                # Check if row already exists
                                                exists = False
                                                for _, existing_row in all_tables[next_table].iterrows():
                                                    if all(existing_row[k] == v for k, v in row_dict.items() if k in existing_row):
                                                        exists = True
                                                        break
                                                
                                                if not exists:
                                                    all_tables[next_table] = pd.concat([all_tables[next_table], pd.DataFrame([row_dict])], ignore_index=True)
                                                    context_rows_added += 1
                                            
                                            relationships_preserved += 1
                                            break
                                elif rel.get('parent_table') == next_table and rel.get('child_table') == current_table:
                                    # next_table is parent, current_table is child
                                    parent_column = rel.get('parent_column')
                                    child_column = rel.get('child_column')
                                    
                                    if child_column in current_values:
                                        # Find rows in next_table where parent_column matches child_column value
                                        value = current_values[child_column]
                                        related_rows = all_tables[next_table][all_tables[next_table][parent_column] == value]
                                        
                                        if not related_rows.empty:
                                            # Add related rows to the table
                                            current_table = next_table
                                            current_values = related_rows.iloc[0].to_dict()
                                            
                                            # Add rows to the table if not already present
                                            for _, row in related_rows.iterrows():
                                                row_dict = row.to_dict()
                                                
                                                # Check if row already exists
                                                exists = False
                                                for _, existing_row in all_tables[next_table].iterrows():
                                                    if all(existing_row[k] == v for k, v in row_dict.items() if k in existing_row):
                                                        exists = True
                                                        break
                                                
                                                if not exists:
                                                    all_tables[next_table] = pd.concat([all_tables[next_table], pd.DataFrame([row_dict])], ignore_index=True)
                                                    context_rows_added += 1
                                            
                                            relationships_preserved += 1
                                            break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # No path found
                    pass
        
        # Update domain partitions
        integrated_domain_partitions = {}
        for domain, tables in domain_partitions.items():
            integrated_domain_partitions[domain] = {}
            for table_name, df in tables.items():
                if table_name in all_tables:
                    integrated_domain_partitions[domain][table_name] = all_tables[table_name]
                else:
                    integrated_domain_partitions[domain][table_name] = df
        
        # Count integrated rows
        integrated_row_count = sum(len(df) for df in all_tables.values())
        
        # Calculate relationship preservation rate
        total_relationships = len(relationships)
        relationship_preservation_rate = 100.0
        if total_relationships > 0:
            relationship_preservation_rate = (relationships_preserved / total_relationships) * 100.0
        
        # Create integration result
        result = IntegrationResult(
            name="anomaly_context",
            tables=list(all_tables.keys()),
            original_row_count=original_row_count,
            integrated_row_count=integrated_row_count,
            anomaly_count=anomaly_count,
            relationships_preserved=relationships_preserved,
            total_relationships=total_relationships,
            relationship_preservation_rate=relationship_preservation_rate,
            created_at=datetime.now()
        )
        
        # Save result
        self._save_result(result)
        
        self.logger.info(f"Anomaly context integration completed with {context_rows_added} context rows added")
        return integrated_domain_partitions, result
    
    def _save_result(self, result: IntegrationResult) -> None:
        """Save integration result to file.
        
        Args:
            result: IntegrationResult instance.
        """
        # Convert to dictionary
        result_dict = result.to_dict()
        
        # Save to file
        output_file = os.path.join(self.output_dir, f"{result.name}_result.json")
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Integration result saved to {output_file}")


class PurposeSpecificIntegrator(Component):
    """Component for creating purpose-specific integrated test datasets."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the purpose-specific integrator.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.output_dir = None
        self.purpose_config = None
    
    def initialize(self) -> None:
        """Initialize the purpose-specific integrator.
        
        Raises:
            ConfigurationError: If the integrator cannot be initialized.
        """
        # Create output directory
        self.output_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'data_integration',
            'purpose_specific'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get configuration
        self.purpose_config = self.config_manager.get('data_integration.purpose_specific', {})
        
        self.logger.info("Purpose-specific integrator initialized")
    
    def validate(self) -> bool:
        """Validate the purpose-specific integrator configuration and state.
        
        Returns:
            True if the integrator is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if not os.path.exists(self.output_dir):
            raise ValidationError(f"Output directory does not exist: {self.output_dir}")
        
        return True
    
    def integrate(self, domain_partitions: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Create purpose-specific integrated test datasets.
        
        Args:
            domain_partitions: Dictionary mapping domain names to dictionaries of table DataFrames.
        
        Returns:
            Dictionary mapping purpose names to dictionaries of table DataFrames.
        """
        self.logger.info("Creating purpose-specific integrated test datasets")
        
        # Get purpose definitions
        purposes = self.purpose_config.get('purposes', [])
        
        if not purposes:
            self.logger.warning("No purpose definitions found in configuration")
            return {}
        
        # Create purpose-specific datasets
        purpose_datasets = {}
        
        for purpose in purposes:
            purpose_name = purpose.get('name')
            included_domains = purpose.get('domains', [])
            included_tables = purpose.get('tables', [])
            
            if not purpose_name:
                continue
            
            self.logger.info(f"Creating dataset for purpose '{purpose_name}'")
            
            # Create dataset for this purpose
            purpose_dataset = {}
            
            # Include domains
            for domain in included_domains:
                if domain in domain_partitions:
                    for table_name, df in domain_partitions[domain].items():
                        purpose_dataset[table_name] = df
            
            # Include specific tables
            for table_spec in included_tables:
                domain = table_spec.get('domain')
                table_name = table_spec.get('table')
                
                if domain and table_name and domain in domain_partitions and table_name in domain_partitions[domain]:
                    purpose_dataset[table_name] = domain_partitions[domain][table_name]
            
            # Save dataset
            if purpose_dataset:
                purpose_datasets[purpose_name] = purpose_dataset
                
                # Save result
                self._save_result(purpose_name, purpose_dataset)
        
        self.logger.info(f"Created {len(purpose_datasets)} purpose-specific datasets")
        return purpose_datasets
    
    def _save_result(self, purpose_name: str, dataset: Dict[str, pd.DataFrame]) -> None:
        """Save purpose-specific dataset result to file.
        
        Args:
            purpose_name: Name of the purpose.
            dataset: Dictionary mapping table names to DataFrames.
        """
        # Create result
        result = {
            'name': purpose_name,
            'tables': list(dataset.keys()),
            'row_counts': {table_name: len(df) for table_name, df in dataset.items()},
            'total_rows': sum(len(df) for df in dataset.values()),
            'created_at': datetime.now().isoformat()
        }
        
        # Save to file
        output_file = os.path.join(self.output_dir, f"{purpose_name}_result.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        self.logger.info(f"Purpose-specific dataset result saved to {output_file}")


class DatasetExporter(Component):
    """Component for exporting integrated datasets."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the dataset exporter.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.output_dir = None
        self.export_config = None
    
    def initialize(self) -> None:
        """Initialize the dataset exporter.
        
        Raises:
            ConfigurationError: If the exporter cannot be initialized.
        """
        # Create output directory
        self.output_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'data_integration',
            'exports'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get configuration
        self.export_config = self.config_manager.get('data_integration.export', {})
        
        self.logger.info("Dataset exporter initialized")
    
    def validate(self) -> bool:
        """Validate the dataset exporter configuration and state.
        
        Returns:
            True if the exporter is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if not os.path.exists(self.output_dir):
            raise ValidationError(f"Output directory does not exist: {self.output_dir}")
        
        return True
    
    def export_datasets(self, datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        """Export datasets to files.
        
        Args:
            datasets: Dictionary mapping dataset names to dictionaries of table DataFrames.
        
        Returns:
            Dictionary mapping dataset names to dictionaries of table file paths.
        """
        self.logger.info(f"Exporting {len(datasets)} datasets")
        
        # Get export format
        export_format = self.export_config.get('format', 'csv')
        
        # Export datasets
        export_paths = {}
        
        for dataset_name, tables in datasets.items():
            # Create directory for this dataset
            dataset_dir = os.path.join(self.output_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Export tables
            table_paths = {}
            
            for table_name, df in tables.items():
                # Export table
                if export_format == 'csv':
                    file_path = os.path.join(dataset_dir, f"{table_name}.csv")
                    df.to_csv(file_path, index=False)
                elif export_format == 'parquet':
                    file_path = os.path.join(dataset_dir, f"{table_name}.parquet")
                    df.to_parquet(file_path, index=False)
                elif export_format == 'json':
                    file_path = os.path.join(dataset_dir, f"{table_name}.json")
                    df.to_json(file_path, orient='records', lines=True)
                else:
                    # Default to CSV
                    file_path = os.path.join(dataset_dir, f"{table_name}.csv")
                    df.to_csv(file_path, index=False)
                
                table_paths[table_name] = file_path
            
            export_paths[dataset_name] = table_paths
        
        self.logger.info(f"Exported {len(datasets)} datasets to {self.output_dir}")
        return export_paths


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
        self.purpose_specific_integrator = None
        self.dataset_exporter = None
    
    def initialize(self) -> None:
        """Initialize the pipeline.
        
        Raises:
            ConfigurationError: If the pipeline cannot be initialized.
        """
        # Initialize components
        self.referential_integrity_integrator = ReferentialIntegrityIntegrator(self.config_manager)
        self.referential_integrity_integrator.initialize()
        
        self.anomaly_context_integrator = AnomalyContextIntegrator(self.config_manager)
        self.anomaly_context_integrator.initialize()
        
        self.purpose_specific_integrator = PurposeSpecificIntegrator(self.config_manager)
        self.purpose_specific_integrator.initialize()
        
        self.dataset_exporter = DatasetExporter(self.config_manager)
        self.dataset_exporter.initialize()
        
        # Add pipeline steps
        self.add_step(ReferentialIntegrityIntegrationStep(self.config_manager, self.referential_integrity_integrator))
        self.add_step(AnomalyContextIntegrationStep(self.config_manager, self.anomaly_context_integrator))
        self.add_step(PurposeSpecificIntegrationStep(self.config_manager, self.purpose_specific_integrator))
        self.add_step(DatasetExportStep(self.config_manager, self.dataset_exporter))
        
        self.logger.info("Data integration pipeline initialized")
    
    def validate(self) -> bool:
        """Validate the pipeline configuration and state.
        
        Returns:
            True if the pipeline is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        # Validate components
        self.referential_integrity_integrator.validate()
        self.anomaly_context_integrator.validate()
        self.purpose_specific_integrator.validate()
        self.dataset_exporter.validate()
        
        return True


class ReferentialIntegrityIntegrationStep(PipelineStep):
    """Pipeline step for integrating data with referential integrity."""
    
    def __init__(self, config_manager: ConfigManager, integrator: ReferentialIntegrityIntegrator):
        """Initialize the referential integrity integration step.
        
        Args:
            config_manager: Configuration manager instance.
            integrator: ReferentialIntegrityIntegrator instance.
        """
        super().__init__(config_manager)
        self.integrator = integrator
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the referential integrity integration step.
        
        Args:
            input_data: Dictionary with data, relationships, domain partitions, anomalies, and sampling results.
        
        Returns:
            Dictionary with input data and integrated domain partitions.
        
        Raises:
            ProcessingError: If integration fails.
        """
        self.logger.info("Integrating data with referential integrity")
        
        if not input_data or 'domain_partitions' not in input_data:
            raise ProcessingError("No domain partitions to integrate")
        
        # Check if referential integrity integration is enabled
        if not self.config_manager.get('data_integration.referential_integrity.enabled', True):
            self.logger.info("Referential integrity integration is disabled in configuration")
            return {**input_data, 'integration_results': []}
        
        # Get domain partitions
        domain_partitions = input_data['domain_partitions']
        
        # Get relationships
        relationships = input_data.get('relationships', [])
        
        # Integrate data
        try:
            integrated_domain_partitions, result = self.integrator.integrate(domain_partitions, relationships)
            
            self.logger.info(f"Referential integrity integration completed with {result.relationships_preserved}/{result.total_relationships} relationships preserved")
            
            return {
                **input_data,
                'domain_partitions': integrated_domain_partitions,
                'integration_results': [result]
            }
        except Exception as e:
            self.logger.error(f"Error integrating data with referential integrity: {str(e)}")
            return {**input_data, 'integration_results': []}


class AnomalyContextIntegrationStep(PipelineStep):
    """Pipeline step for integrating anomalies with their context."""
    
    def __init__(self, config_manager: ConfigManager, integrator: AnomalyContextIntegrator):
        """Initialize the anomaly context integration step.
        
        Args:
            config_manager: Configuration manager instance.
            integrator: AnomalyContextIntegrator instance.
        """
        super().__init__(config_manager)
        self.integrator = integrator
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the anomaly context integration step.
        
        Args:
            input_data: Dictionary with data, relationships, domain partitions, anomalies, and integration results.
        
        Returns:
            Dictionary with input data and updated integrated domain partitions.
        
        Raises:
            ProcessingError: If integration fails.
        """
        self.logger.info("Integrating anomalies with their context")
        
        if not input_data or 'domain_partitions' not in input_data:
            raise ProcessingError("No domain partitions to integrate")
        
        # Check if anomaly context integration is enabled
        if not self.config_manager.get('data_integration.anomaly_context.enabled', True):
            self.logger.info("Anomaly context integration is disabled in configuration")
            return input_data
        
        # Get domain partitions
        domain_partitions = input_data['domain_partitions']
        
        # Get anomalies
        anomalies = input_data.get('anomalies', [])
        
        if not anomalies:
            self.logger.info("No anomalies to integrate")
            return input_data
        
        # Get relationships
        relationships = input_data.get('relationships', [])
        
        # Get existing integration results
        integration_results = input_data.get('integration_results', [])
        
        # Integrate anomalies
        try:
            integrated_domain_partitions, result = self.integrator.integrate(domain_partitions, anomalies, relationships)
            
            self.logger.info(f"Anomaly context integration completed with {result.integrated_row_count - result.original_row_count} context rows added")
            
            # Add result to integration results
            integration_results.append(result)
            
            return {
                **input_data,
                'domain_partitions': integrated_domain_partitions,
                'integration_results': integration_results
            }
        except Exception as e:
            self.logger.error(f"Error integrating anomalies with their context: {str(e)}")
            return input_data


class PurposeSpecificIntegrationStep(PipelineStep):
    """Pipeline step for creating purpose-specific integrated test datasets."""
    
    def __init__(self, config_manager: ConfigManager, integrator: PurposeSpecificIntegrator):
        """Initialize the purpose-specific integration step.
        
        Args:
            config_manager: Configuration manager instance.
            integrator: PurposeSpecificIntegrator instance.
        """
        super().__init__(config_manager)
        self.integrator = integrator
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the purpose-specific integration step.
        
        Args:
            input_data: Dictionary with data, relationships, domain partitions, anomalies, and integration results.
        
        Returns:
            Dictionary with input data and purpose-specific datasets.
        
        Raises:
            ProcessingError: If integration fails.
        """
        self.logger.info("Creating purpose-specific integrated test datasets")
        
        if not input_data or 'domain_partitions' not in input_data:
            raise ProcessingError("No domain partitions to integrate")
        
        # Check if purpose-specific integration is enabled
        if not self.config_manager.get('data_integration.purpose_specific.enabled', True):
            self.logger.info("Purpose-specific integration is disabled in configuration")
            return input_data
        
        # Get domain partitions
        domain_partitions = input_data['domain_partitions']
        
        # Create purpose-specific datasets
        try:
            purpose_datasets = self.integrator.integrate(domain_partitions)
            
            self.logger.info(f"Created {len(purpose_datasets)} purpose-specific datasets")
            
            return {
                **input_data,
                'purpose_datasets': purpose_datasets
            }
        except Exception as e:
            self.logger.error(f"Error creating purpose-specific datasets: {str(e)}")
            return input_data


class DatasetExportStep(PipelineStep):
    """Pipeline step for exporting integrated datasets."""
    
    def __init__(self, config_manager: ConfigManager, exporter: DatasetExporter):
        """Initialize the dataset export step.
        
        Args:
            config_manager: Configuration manager instance.
            exporter: DatasetExporter instance.
        """
        super().__init__(config_manager)
        self.exporter = exporter
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the dataset export step.
        
        Args:
            input_data: Dictionary with data, relationships, domain partitions, purpose datasets, and integration results.
        
        Returns:
            Dictionary with input data and export paths.
        
        Raises:
            ProcessingError: If export fails.
        """
        self.logger.info("Exporting integrated datasets")
        
        # Check if export is enabled
        if not self.config_manager.get('data_integration.export.enabled', True):
            self.logger.info("Dataset export is disabled in configuration")
            return input_data
        
        # Get datasets to export
        datasets = {}
        
        # Add domain partitions
        if 'domain_partitions' in input_data:
            for domain, tables in input_data['domain_partitions'].items():
                datasets[f"domain_{domain}"] = tables
        
        # Add purpose-specific datasets
        if 'purpose_datasets' in input_data:
            datasets.update(input_data['purpose_datasets'])
        
        if not datasets:
            self.logger.info("No datasets to export")
            return input_data
        
        # Export datasets
        try:
            export_paths = self.exporter.export_datasets(datasets)
            
            self.logger.info(f"Exported {len(export_paths)} datasets")
            
            return {
                **input_data,
                'export_paths': export_paths
            }
        except Exception as e:
            self.logger.error(f"Error exporting datasets: {str(e)}")
            return input_data

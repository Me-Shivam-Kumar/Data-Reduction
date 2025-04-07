"""
Data Classification and Partitioning module for the Data Warehouse Subsampling Framework.

This module provides components for classifying and partitioning data
across business domains, identifying relationships between tables,
and preparing data for subsequent processing steps.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import re
from collections import defaultdict
import networkx as nx

from ..common.base import Component, ConfigManager, Pipeline, PipelineStep, Relationship, ProcessingError
from ..common.utils import save_dataframe, load_dataframe, save_json, load_json, ensure_directory
from ..common.connectors import DataConnector, create_connector

logger = logging.getLogger(__name__)


class DomainDetector(Component):
    """Component for detecting and classifying data domains."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the domain detector.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.method = None
        self.manual_domains = None
    
    def initialize(self) -> None:
        """Initialize the domain detector.
        
        Raises:
            ConfigurationError: If the domain detector cannot be initialized.
        """
        self.method = self.config_manager.get('data_classification.domain_detection.method', 'rule_based')
        self.manual_domains = self.config_manager.get('data_classification.domain_detection.manual_domains', {})
        
        self.logger.info(f"Domain detector initialized with method: {self.method}")
    
    def detect_domains(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Detect domains in the data.
        
        This method analyzes table structures, column names, and data patterns
        to group related tables into logical domains.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
        
        Returns:
            Dictionary mapping domain names to lists of table names.
        """
        self.logger.info(f"Detecting domains using method: {self.method}")
        
        if self.method == 'manual':
            return self._detect_domains_manual(tables)
        elif self.method == 'rule_based':
            return self._detect_domains_rule_based(tables)
        elif self.method == 'clustering':
            return self._detect_domains_clustering(tables)
        else:
            self.logger.warning(f"Unknown domain detection method: {self.method}, falling back to rule-based")
            return self._detect_domains_rule_based(tables)
    
    def _detect_domains_manual(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Detect domains using manual configuration.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
        
        Returns:
            Dictionary mapping domain names to lists of table names.
        """
        self.logger.info("Detecting domains using manual configuration")
        
        # Validate manual domains
        for domain, domain_tables in self.manual_domains.items():
            for table in domain_tables:
                if table not in tables:
                    self.logger.warning(f"Table {table} in domain {domain} not found in data")
        
        # Assign unassigned tables to 'other' domain
        assigned_tables = set()
        for domain_tables in self.manual_domains.values():
            assigned_tables.update(domain_tables)
        
        unassigned_tables = set(tables.keys()) - assigned_tables
        
        domains = {domain: list(domain_tables) for domain, domain_tables in self.manual_domains.items()}
        
        if unassigned_tables:
            domains['other'] = list(unassigned_tables)
        
        return domains
    
    def _detect_domains_rule_based(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Detect domains using rule-based approach.
        
        This method uses table name patterns and column relationships
        to group tables into domains.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
        
        Returns:
            Dictionary mapping domain names to lists of table names.
        """
        self.logger.info("Detecting domains using rule-based approach")
        
        # Define common domain prefixes/patterns
        domain_patterns = {
            'customer': [r'customer', r'client', r'account'],
            'product': [r'product', r'item', r'catalog'],
            'order': [r'order', r'sale', r'transaction'],
            'inventory': [r'inventory', r'stock', r'warehouse'],
            'employee': [r'employee', r'staff', r'personnel'],
            'finance': [r'finance', r'accounting', r'payment'],
            'marketing': [r'marketing', r'campaign', r'promotion'],
            'shipping': [r'shipping', r'delivery', r'logistics']
        }
        
        # Compile patterns
        compiled_patterns = {
            domain: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for domain, patterns in domain_patterns.items()
        }
        
        # Assign tables to domains
        table_domains = {}
        for table_name in tables.keys():
            assigned = False
            
            for domain, patterns in compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(table_name):
                        table_domains[table_name] = domain
                        assigned = True
                        break
                
                if assigned:
                    break
            
            if not assigned:
                table_domains[table_name] = 'other'
        
        # Group tables by domain
        domains = defaultdict(list)
        for table_name, domain in table_domains.items():
            domains[domain].append(table_name)
        
        return dict(domains)
    
    def _detect_domains_clustering(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Detect domains using clustering approach.
        
        This method uses column name similarity and data patterns
        to cluster tables into domains.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
        
        Returns:
            Dictionary mapping domain names to lists of table names.
        """
        self.logger.info("Detecting domains using clustering approach")
        
        # Create a graph of table relationships
        G = nx.Graph()
        
        # Add nodes for each table
        for table_name in tables.keys():
            G.add_node(table_name)
        
        # Add edges based on column name similarity
        for table1, df1 in tables.items():
            cols1 = set(df1.columns)
            
            for table2, df2 in tables.items():
                if table1 >= table2:  # Avoid duplicate edges and self-loops
                    continue
                
                cols2 = set(df2.columns)
                
                # Calculate Jaccard similarity of column names
                intersection = len(cols1.intersection(cols2))
                union = len(cols1.union(cols2))
                
                if union > 0:
                    similarity = intersection / union
                    
                    # Add edge if similarity is above threshold
                    if similarity > 0.2:
                        G.add_edge(table1, table2, weight=similarity)
        
        # Find communities (domains) using Louvain method
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G)
            
            # Group tables by community
            domains = defaultdict(list)
            for table_name, community_id in partition.items():
                domains[f"domain_{community_id}"].append(table_name)
            
            return dict(domains)
        except ImportError:
            self.logger.warning("community package not found, falling back to connected components")
            
            # Find connected components
            components = list(nx.connected_components(G))
            
            # Group tables by component
            domains = {}
            for i, component in enumerate(components):
                domains[f"domain_{i}"] = list(component)
            
            return domains


class RelationshipDiscoverer(Component):
    """Component for discovering relationships between tables."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the relationship discoverer.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.methods = None
        self.min_confidence = None
    
    def initialize(self) -> None:
        """Initialize the relationship discoverer.
        
        Raises:
            ConfigurationError: If the relationship discoverer cannot be initialized.
        """
        self.methods = self.config_manager.get('data_classification.relationship_discovery.methods', ['name_matching', 'foreign_key_analysis'])
        self.min_confidence = self.config_manager.get('data_classification.relationship_discovery.min_confidence', 0.8)
        
        self.logger.info(f"Relationship discoverer initialized with methods: {self.methods}")
    
    def discover_relationships(self, tables: Dict[str, pd.DataFrame]) -> List[Relationship]:
        """Discover relationships between tables.
        
        This method analyzes foreign key relationships, naming patterns,
        and data correlations to identify relationships between tables.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
        
        Returns:
            List of discovered relationships.
        """
        self.logger.info(f"Discovering relationships using methods: {self.methods}")
        
        relationships = []
        
        if 'name_matching' in self.methods:
            relationships.extend(self._discover_relationships_name_matching(tables))
        
        if 'foreign_key_analysis' in self.methods:
            relationships.extend(self._discover_relationships_foreign_key_analysis(tables))
        
        if 'data_profiling' in self.methods:
            relationships.extend(self._discover_relationships_data_profiling(tables))
        
        # Filter relationships by confidence
        relationships = [r for r in relationships if r.confidence >= self.min_confidence]
        
        # Remove duplicates
        unique_relationships = []
        relationship_keys = set()
        
        for r in relationships:
            key = (r.parent_table, r.parent_column, r.child_table, r.child_column)
            
            if key not in relationship_keys:
                relationship_keys.add(key)
                unique_relationships.append(r)
        
        self.logger.info(f"Discovered {len(unique_relationships)} unique relationships")
        return unique_relationships
    
    def _discover_relationships_name_matching(self, tables: Dict[str, pd.DataFrame]) -> List[Relationship]:
        """Discover relationships using name matching.
        
        This method identifies potential relationships based on column naming patterns.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
        
        Returns:
            List of discovered relationships.
        """
        self.logger.info("Discovering relationships using name matching")
        
        relationships = []
        
        for parent_table, parent_df in tables.items():
            for parent_column in parent_df.columns:
                # Check if column name ends with '_id' or is 'id'
                if parent_column.lower() == 'id' or parent_column.lower().endswith('_id'):
                    # For 'id' column, look for tables that might reference this table
                    if parent_column.lower() == 'id':
                        for child_table, child_df in tables.items():
                            if child_table == parent_table:
                                continue
                            
                            # Look for columns like 'parent_table_id'
                            expected_column = f"{parent_table.lower().rstrip('s')}_id"
                            
                            for child_column in child_df.columns:
                                if child_column.lower() == expected_column:
                                    relationships.append(Relationship(
                                        parent_table=parent_table,
                                        parent_column=parent_column,
                                        child_table=child_table,
                                        child_column=child_column,
                                        confidence=0.9
                                    ))
                    
                    # For 'xxx_id' columns, look for tables with 'id' column
                    elif parent_column.lower().endswith('_id'):
                        # Extract the referenced table name
                        referenced_table = parent_column.lower()[:-3]
                        
                        # Handle common plural forms
                        if referenced_table.endswith('s'):
                            referenced_table_singular = referenced_table[:-1]
                        else:
                            referenced_table_singular = referenced_table
                        
                        # Look for matching tables
                        for child_table, child_df in tables.items():
                            if child_table.lower() == referenced_table or child_table.lower() == referenced_table_singular:
                                if 'id' in child_df.columns:
                                    relationships.append(Relationship(
                                        parent_table=child_table,
                                        parent_column='id',
                                        child_table=parent_table,
                                        child_column=parent_column,
                                        confidence=0.9
                                    ))
        
        return relationships
    
    def _discover_relationships_foreign_key_analysis(self, tables: Dict[str, pd.DataFrame]) -> List[Relationship]:
        """Discover relationships using foreign key analysis.
        
        This method analyzes data patterns to identify potential foreign key relationships.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
        
        Returns:
            List of discovered relationships.
        """
        self.logger.info("Discovering relationships using foreign key analysis")
        
        relationships = []
        
        for parent_table, parent_df in tables.items():
            # Identify potential primary key columns
            potential_pk_columns = []
            
            for column in parent_df.columns:
                # Check if column has unique values
                if parent_df[column].nunique() == len(parent_df) and not parent_df[column].isna().any():
                    potential_pk_columns.append(column)
            
            if not potential_pk_columns:
                continue
            
            # For each potential primary key, look for matching foreign keys
            for pk_column in potential_pk_columns:
                pk_values = set(parent_df[pk_column])
                
                for child_table, child_df in tables.items():
                    if child_table == parent_table:
                        continue
                    
                    for child_column in child_df.columns:
                        # Skip if column has all unique values (likely a primary key)
                        if child_df[child_column].nunique() == len(child_df) and not child_df[child_column].isna().any():
                            continue
                        
                        # Check if child column values are a subset of parent column values
                        child_values = set(child_df[child_column].dropna())
                        
                        if not child_values:
                            continue
                        
                        if child_values.issubset(pk_values):
                            # Calculate confidence based on coverage
                            coverage = len(child_values) / len(pk_values) if pk_values else 0
                            confidence = min(0.8 + coverage * 0.2, 1.0)
                            
                            relationships.append(Relationship(
                                parent_table=parent_table,
                                parent_column=pk_column,
                                child_table=child_table,
                                child_column=child_column,
                                confidence=confidence
                            ))
        
        return relationships
    
    def _discover_relationships_data_profiling(self, tables: Dict[str, pd.DataFrame]) -> List[Relationship]:
        """Discover relationships using data profiling.
        
        This method analyzes data distributions and patterns to identify potential relationships.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
        
        Returns:
            List of discovered relationships.
        """
        self.logger.info("Discovering relationships using data profiling")
        
        relationships = []
        
        # This is a simplified implementation
        # A more comprehensive implementation would use statistical methods
        # to analyze data distributions and patterns
        
        for parent_table, parent_df in tables.items():
            for parent_column in parent_df.columns:
                parent_values = parent_df[parent_column].dropna()
                
                # Skip columns with too many unique values (likely not a key)
                if parent_values.nunique() > len(parent_df) * 0.9:
                    continue
                
                for child_table, child_df in tables.items():
                    if child_table == parent_table:
                        continue
                    
                    for child_column in child_df.columns:
                        child_values = child_df[child_column].dropna()
                        
                        # Skip columns with too many unique values (likely not a foreign key)
                        if child_values.nunique() > len(child_df) * 0.9:
                            continue
                        
                        # Check if data types are compatible
                        if parent_values.dtype != child_values.dtype:
                            continue
                        
                        # Check if there's significant overlap in values
                        parent_set = set(parent_values)
                        child_set = set(child_values)
                        
                        if not parent_set or not child_set:
                            continue
                        
                        overlap = len(parent_set.intersection(child_set))
                        
                        if overlap > 0:
                            # Calculate confidence based on overlap
                            overlap_ratio = overlap / len(child_set)
                            
                            if overlap_ratio > 0.8:
                                confidence = 0.7 + overlap_ratio * 0.3
                                
                                relationships.append(Relationship(
                                    parent_table=parent_table,
                                    parent_column=parent_column,
                                    child_table=child_table,
                                    child_column=child_column,
                                    confidence=confidence
                                ))
        
        return relationships


class DataPartitioner(Component):
    """Component for partitioning data into domains."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the data partitioner.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.method = None
        self.max_partition_size_gb = None
    
    def initialize(self) -> None:
        """Initialize the data partitioner.
        
        Raises:
            ConfigurationError: If the data partitioner cannot be initialized.
        """
        self.method = self.config_manager.get('data_classification.partitioning.method', 'domain_based')
        self.max_partition_size_gb = self.config_manager.get('data_classification.partitioning.max_partition_size_gb', 10)
        
        self.logger.info(f"Data partitioner initialized with method: {self.method}")
    
    def partition_data(self, tables: Dict[str, pd.DataFrame], domains: Dict[str, List[str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Partition data into domains.
        
        This method organizes tables into domain-specific partitions
        for more efficient processing in subsequent steps.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
            domains: Dictionary mapping domain names to lists of table names.
        
        Returns:
            Dictionary mapping domain names to dictionaries mapping table names to DataFrames.
        """
        self.logger.info(f"Partitioning data using method: {self.method}")
        
        if self.method == 'domain_based':
            return self._partition_data_domain_based(tables, domains)
        elif self.method == 'size_based':
            return self._partition_data_size_based(tables, domains)
        elif self.method == 'hybrid':
            return self._partition_data_hybrid(tables, domains)
        else:
            self.logger.warning(f"Unknown partitioning method: {self.method}, falling back to domain-based")
            return self._partition_data_domain_based(tables, domains)
    
    def _partition_data_domain_based(self, tables: Dict[str, pd.DataFrame], domains: Dict[str, List[str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Partition data based on domains.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
            domains: Dictionary mapping domain names to lists of table names.
        
        Returns:
            Dictionary mapping domain names to dictionaries mapping table names to DataFrames.
        """
        self.logger.info("Partitioning data based on domains")
        
        partitions = {}
        
        for domain, domain_tables in domains.items():
            partition = {}
            
            for table_name in domain_tables:
                if table_name in tables:
                    partition[table_name] = tables[table_name]
            
            if partition:
                partitions[domain] = partition
        
        return partitions
    
    def _partition_data_size_based(self, tables: Dict[str, pd.DataFrame], domains: Dict[str, List[str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Partition data based on size.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
            domains: Dictionary mapping domain names to lists of table names.
        
        Returns:
            Dictionary mapping partition names to dictionaries mapping table names to DataFrames.
        """
        self.logger.info("Partitioning data based on size")
        
        # Calculate table sizes
        table_sizes = {}
        for table_name, df in tables.items():
            # Estimate size in GB
            size_gb = df.memory_usage(deep=True).sum() / (1024 ** 3)
            table_sizes[table_name] = size_gb
        
        # Sort tables by size (largest first)
        sorted_tables = sorted(table_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Create partitions
        partitions = {}
        current_partition = {}
        current_partition_size = 0
        partition_index = 0
        
        for table_name, size_gb in sorted_tables:
            # If table is larger than max partition size, create a separate partition
            if size_gb > self.max_partition_size_gb:
                partitions[f"partition_{partition_index}"] = {table_name: tables[table_name]}
                partition_index += 1
                continue
            
            # If adding table would exceed max partition size, create a new partition
            if current_partition_size + size_gb > self.max_partition_size_gb and current_partition:
                partitions[f"partition_{partition_index}"] = current_partition
                current_partition = {}
                current_partition_size = 0
                partition_index += 1
            
            # Add table to current partition
            current_partition[table_name] = tables[table_name]
            current_partition_size += size_gb
        
        # Add final partition if not empty
        if current_partition:
            partitions[f"partition_{partition_index}"] = current_partition
        
        return partitions
    
    def _partition_data_hybrid(self, tables: Dict[str, pd.DataFrame], domains: Dict[str, List[str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Partition data using a hybrid approach.
        
        This method first partitions data by domain, then splits large domains
        into smaller partitions based on size.
        
        Args:
            tables: Dictionary mapping table names to DataFrames.
            domains: Dictionary mapping domain names to lists of table names.
        
        Returns:
            Dictionary mapping partition names to dictionaries mapping table names to DataFrames.
        """
        self.logger.info("Partitioning data using hybrid approach")
        
        # First, partition by domain
        domain_partitions = self._partition_data_domain_based(tables, domains)
        
        # Then, split large domains
        partitions = {}
        
        for domain, domain_tables in domain_partitions.items():
            # Calculate domain size
            domain_size_gb = sum(df.memory_usage(deep=True).sum() / (1024 ** 3) for df in domain_tables.values())
            
            # If domain is small enough, keep it as is
            if domain_size_gb <= self.max_partition_size_gb:
                partitions[domain] = domain_tables
                continue
            
            # Otherwise, split domain based on size
            table_sizes = {}
            for table_name, df in domain_tables.items():
                size_gb = df.memory_usage(deep=True).sum() / (1024 ** 3)
                table_sizes[table_name] = size_gb
            
            # Sort tables by size (largest first)
            sorted_tables = sorted(table_sizes.items(), key=lambda x: x[1], reverse=True)
            
            # Create partitions
            current_partition = {}
            current_partition_size = 0
            partition_index = 0
            
            for table_name, size_gb in sorted_tables:
                # If table is larger than max partition size, create a separate partition
                if size_gb > self.max_partition_size_gb:
                    partitions[f"{domain}_{partition_index}"] = {table_name: domain_tables[table_name]}
                    partition_index += 1
                    continue
                
                # If adding table would exceed max partition size, create a new partition
                if current_partition_size + size_gb > self.max_partition_size_gb and current_partition:
                    partitions[f"{domain}_{partition_index}"] = current_partition
                    current_partition = {}
                    current_partition_size = 0
                    partition_index += 1
                
                # Add table to current partition
                current_partition[table_name] = domain_tables[table_name]
                current_partition_size += size_gb
            
            # Add final partition if not empty
            if current_partition:
                partitions[f"{domain}_{partition_index}"] = current_partition
        
        return partitions


class DataClassificationPipeline(Pipeline):
    """Pipeline for data classification and partitioning."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the data classification pipeline.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.domain_detector = None
        self.relationship_discoverer = None
        self.data_partitioner = None
        self.connector = None
    
    def initialize(self) -> None:
        """Initialize the data classification pipeline.
        
        Raises:
            ConfigurationError: If the pipeline cannot be initialized.
        """
        super().initialize()
        
        # Initialize components
        self.domain_detector = DomainDetector(self.config_manager)
        self.domain_detector.initialize()
        
        self.relationship_discoverer = RelationshipDiscoverer(self.config_manager)
        self.relationship_discoverer.initialize()
        
        self.data_partitioner = DataPartitioner(self.config_manager)
        self.data_partitioner.initialize()
        
        # Create data connector
        self.connector = create_connector(self.config_manager)
        
        # Initialize steps
        self.steps = [
            PipelineStep(self.load_data),
            PipelineStep(self.detect_domains),
            PipelineStep(self.discover_relationships),
            PipelineStep(self.partition_data),
            PipelineStep(self.save_results)
        ]
        
        self.logger.info("Data classification pipeline initialized")
    
    def load_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load data from the data source.
        
        Args:
            data: Input data.
        
        Returns:
            Updated data with loaded tables.
        
        Raises:
            ProcessingError: If data loading fails.
        """
        self.logger.info("Loading data from data source")
        
        try:
            # Connect to data source
            self.connector.connect()
            
            # Get list of tables
            table_list = self.connector.get_tables()
            
            # Filter tables based on configuration
            include_tables = self.config_manager.get('data_sources.primary_warehouse.tables.include', [])
            exclude_tables = self.config_manager.get('data_sources.primary_warehouse.tables.exclude', [])
            
            if include_tables:
                table_list = [t for t in table_list if t in include_tables]
            
            if exclude_tables:
                table_list = [t for t in table_list if t not in exclude_tables]
            
            # Load tables
            tables = {}
            for table_name in table_list:
                self.logger.info(f"Loading table: {table_name}")
                tables[table_name] = self.connector.get_table_data(table_name)
            
            # Disconnect from data source
            self.connector.disconnect()
            
            # Update data
            result = data.copy()
            result['tables'] = tables
            result['original_data'] = tables.copy()
            
            self.logger.info(f"Loaded {len(tables)} tables")
            return result
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise ProcessingError(f"Error loading data: {str(e)}")
    
    def detect_domains(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect domains in the data.
        
        Args:
            data: Input data with loaded tables.
        
        Returns:
            Updated data with detected domains.
        
        Raises:
            ProcessingError: If domain detection fails.
        """
        self.logger.info("Detecting domains")
        
        try:
            tables = data.get('tables', {})
            
            if not tables:
                self.logger.warning("No tables found for domain detection")
                return data
            
            # Detect domains
            domains = self.domain_detector.detect_domains(tables)
            
            # Update data
            result = data.copy()
            result['domains'] = domains
            
            self.logger.info(f"Detected {len(domains)} domains")
            return result
        except Exception as e:
            self.logger.error(f"Error detecting domains: {str(e)}")
            raise ProcessingError(f"Error detecting domains: {str(e)}")
    
    def discover_relationships(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Discover relationships between tables.
        
        Args:
            data: Input data with loaded tables and detected domains.
        
        Returns:
            Updated data with discovered relationships.
        
        Raises:
            ProcessingError: If relationship discovery fails.
        """
        self.logger.info("Discovering relationships")
        
        try:
            tables = data.get('tables', {})
            
            if not tables:
                self.logger.warning("No tables found for relationship discovery")
                return data
            
            # Discover relationships
            relationships = self.relationship_discoverer.discover_relationships(tables)
            
            # Update data
            result = data.copy()
            result['relationships'] = relationships
            
            self.logger.info(f"Discovered {len(relationships)} relationships")
            return result
        except Exception as e:
            self.logger.error(f"Error discovering relationships: {str(e)}")
            raise ProcessingError(f"Error discovering relationships: {str(e)}")
    
    def partition_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Partition data into domains.
        
        Args:
            data: Input data with loaded tables, detected domains, and discovered relationships.
        
        Returns:
            Updated data with partitioned data.
        
        Raises:
            ProcessingError: If data partitioning fails.
        """
        self.logger.info("Partitioning data")
        
        try:
            tables = data.get('tables', {})
            domains = data.get('domains', {})
            
            if not tables:
                self.logger.warning("No tables found for data partitioning")
                return data
            
            if not domains:
                self.logger.warning("No domains found for data partitioning")
                return data
            
            # Partition data
            domain_partitions = self.data_partitioner.partition_data(tables, domains)
            
            # Update data
            result = data.copy()
            result['domain_partitions'] = domain_partitions
            
            self.logger.info(f"Created {len(domain_partitions)} partitions")
            return result
        except Exception as e:
            self.logger.error(f"Error partitioning data: {str(e)}")
            raise ProcessingError(f"Error partitioning data: {str(e)}")
    
    def save_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save classification results.
        
        Args:
            data: Input data with all classification results.
        
        Returns:
            Updated data with saved results.
        
        Raises:
            ProcessingError: If saving results fails.
        """
        self.logger.info("Saving classification results")
        
        try:
            # Create output directories
            domains_dir = os.path.join(self.output_dir, 'domains')
            relationships_dir = os.path.join(self.output_dir, 'relationships')
            partitions_dir = os.path.join(self.output_dir, 'partitions')
            
            ensure_directory(domains_dir)
            ensure_directory(relationships_dir)
            ensure_directory(partitions_dir)
            
            # Save domains
            domains = data.get('domains', {})
            if domains:
                domains_file = os.path.join(domains_dir, 'domains.json')
                save_json(domains, domains_file)
            
            # Save relationships
            relationships = data.get('relationships', [])
            if relationships:
                relationships_file = os.path.join(relationships_dir, 'relationships.json')
                save_json([r.to_dict() for r in relationships], relationships_file)
            
            # Save partitions metadata
            domain_partitions = data.get('domain_partitions', {})
            if domain_partitions:
                partitions_metadata = {}
                
                for domain, tables in domain_partitions.items():
                    partitions_metadata[domain] = {
                        'tables': list(tables.keys()),
                        'row_counts': {table: len(df) for table, df in tables.items()},
                        'column_counts': {table: len(df.columns) for table, df in tables.items()}
                    }
                
                partitions_file = os.path.join(partitions_dir, 'partitions.json')
                save_json(partitions_metadata, partitions_file)
            
            # Update data
            result = data.copy()
            result['classification_results'] = {
                'domains_file': os.path.join(domains_dir, 'domains.json'),
                'relationships_file': os.path.join(relationships_dir, 'relationships.json'),
                'partitions_file': os.path.join(partitions_dir, 'partitions.json')
            }
            
            self.logger.info("Classification results saved")
            return result
        except Exception as e:
            self.logger.error(f"Error saving classification results: {str(e)}")
            raise ProcessingError(f"Error saving classification results: {str(e)}")

"""
Anomaly Detection and Isolation module for the Data Warehouse Subsampling Framework.

This module provides components for detecting and isolating anomalies
in the data, ensuring they are preserved for testing purposes.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import json
from collections import defaultdict
from scipy import stats

from ..common.base import Component, ConfigManager, Pipeline, PipelineStep, ProcessingError
from ..common.utils import save_dataframe, load_dataframe, save_json, load_json, ensure_directory

logger = logging.getLogger(__name__)


class AnomalyDetector(Component):
    """Base class for anomaly detectors."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the anomaly detector.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
    
    def detect_anomalies(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect anomalies in a DataFrame.
        
        Args:
            df: DataFrame to analyze.
            table_name: Name of the table.
        
        Returns:
            Tuple of (anomalies DataFrame, normal DataFrame).
        """
        raise NotImplementedError("Subclasses must implement detect_anomalies()")


class StatisticalAnomalyDetector(AnomalyDetector):
    """Anomaly detector using statistical methods."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the statistical anomaly detector.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.methods = None
        self.z_score_threshold = None
        self.iqr_factor = None
    
    def initialize(self) -> None:
        """Initialize the statistical anomaly detector.
        
        Raises:
            ConfigurationError: If the detector cannot be initialized.
        """
        self.methods = self.config_manager.get('anomaly_detection.methods.statistical', {})
        self.z_score_threshold = self.config_manager.get('anomaly_detection.methods.statistical.z_score.threshold', 3.0)
        self.iqr_factor = self.config_manager.get('anomaly_detection.methods.statistical.iqr.factor', 1.5)
        
        self.logger.info(f"Statistical anomaly detector initialized with z-score threshold: {self.z_score_threshold}, IQR factor: {self.iqr_factor}")
    
    def detect_anomalies(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect anomalies using statistical methods.
        
        Args:
            df: DataFrame to analyze.
            table_name: Name of the table.
        
        Returns:
            Tuple of (anomalies DataFrame, normal DataFrame).
        """
        self.logger.info(f"Detecting statistical anomalies in table: {table_name}")
        
        # Create a copy of the DataFrame
        df_copy = df.copy()
        
        # Create a mask for anomalies
        anomaly_mask = pd.Series(False, index=df.index)
        
        # Apply z-score method if enabled
        if self.config_manager.get('anomaly_detection.methods.statistical.z_score.enabled', True):
            z_score_mask = self._detect_z_score_anomalies(df_copy)
            anomaly_mask = anomaly_mask | z_score_mask
        
        # Apply IQR method if enabled
        if self.config_manager.get('anomaly_detection.methods.statistical.iqr.enabled', True):
            iqr_mask = self._detect_iqr_anomalies(df_copy)
            anomaly_mask = anomaly_mask | iqr_mask
        
        # Split data into anomalies and normal
        anomalies_df = df_copy[anomaly_mask].copy()
        normal_df = df_copy[~anomaly_mask].copy()
        
        self.logger.info(f"Detected {len(anomalies_df)} statistical anomalies in table: {table_name}")
        return anomalies_df, normal_df
    
    def _detect_z_score_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detect anomalies using z-score method.
        
        Args:
            df: DataFrame to analyze.
        
        Returns:
            Boolean mask indicating anomalies.
        """
        self.logger.info("Detecting anomalies using z-score method")
        
        # Create a mask for anomalies
        anomaly_mask = pd.Series(False, index=df.index)
        
        # Apply z-score method to numeric columns
        for column in df.select_dtypes(include=np.number).columns:
            # Skip columns with all NaN values
            if df[column].isna().all():
                continue
            
            # Calculate z-scores
            z_scores = np.abs(stats.zscore(df[column].fillna(df[column].mean())))
            
            # Identify anomalies
            column_anomalies = z_scores > self.z_score_threshold
            
            # Update mask
            anomaly_mask = anomaly_mask | column_anomalies
        
        return anomaly_mask
    
    def _detect_iqr_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detect anomalies using IQR method.
        
        Args:
            df: DataFrame to analyze.
        
        Returns:
            Boolean mask indicating anomalies.
        """
        self.logger.info("Detecting anomalies using IQR method")
        
        # Create a mask for anomalies
        anomaly_mask = pd.Series(False, index=df.index)
        
        # Apply IQR method to numeric columns
        for column in df.select_dtypes(include=np.number).columns:
            # Skip columns with all NaN values
            if df[column].isna().all():
                continue
            
            # Calculate IQR
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            
            # Define bounds
            lower_bound = q1 - self.iqr_factor * iqr
            upper_bound = q3 + self.iqr_factor * iqr
            
            # Identify anomalies
            column_anomalies = (df[column] < lower_bound) | (df[column] > upper_bound)
            
            # Update mask
            anomaly_mask = anomaly_mask | column_anomalies
        
        return anomaly_mask


class RuleBasedAnomalyDetector(AnomalyDetector):
    """Anomaly detector using rule-based methods."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the rule-based anomaly detector.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.rules = None
    
    def initialize(self) -> None:
        """Initialize the rule-based anomaly detector.
        
        Raises:
            ConfigurationError: If the detector cannot be initialized.
        """
        self.rules = self.config_manager.get('anomaly_detection.methods.rule_based.rules', [])
        
        self.logger.info(f"Rule-based anomaly detector initialized with {len(self.rules)} rules")
    
    def detect_anomalies(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect anomalies using rule-based methods.
        
        Args:
            df: DataFrame to analyze.
            table_name: Name of the table.
        
        Returns:
            Tuple of (anomalies DataFrame, normal DataFrame).
        """
        self.logger.info(f"Detecting rule-based anomalies in table: {table_name}")
        
        # Create a copy of the DataFrame
        df_copy = df.copy()
        
        # Create a mask for anomalies
        anomaly_mask = pd.Series(False, index=df.index)
        
        # Apply rules for this table
        table_rules = [rule for rule in self.rules if rule.get('table') == table_name]
        
        for rule in table_rules:
            column = rule.get('column')
            condition = rule.get('condition')
            
            if column not in df_copy.columns:
                self.logger.warning(f"Column {column} not found in table {table_name}")
                continue
            
            # Parse and apply condition
            try:
                # Convert condition to a query string
                query = f"{column} {condition}"
                
                # Apply query
                rule_mask = df_copy.eval(query)
                
                # Update mask
                anomaly_mask = anomaly_mask | rule_mask
            except Exception as e:
                self.logger.error(f"Error applying rule {rule}: {str(e)}")
        
        # Split data into anomalies and normal
        anomalies_df = df_copy[anomaly_mask].copy()
        normal_df = df_copy[~anomaly_mask].copy()
        
        self.logger.info(f"Detected {len(anomalies_df)} rule-based anomalies in table: {table_name}")
        return anomalies_df, normal_df


class ClusteringAnomalyDetector(AnomalyDetector):
    """Anomaly detector using clustering methods."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the clustering anomaly detector.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.method = None
        self.contamination = None
    
    def initialize(self) -> None:
        """Initialize the clustering anomaly detector.
        
        Raises:
            ConfigurationError: If the detector cannot be initialized.
        """
        self.method = self.config_manager.get('anomaly_detection.methods.clustering.method', 'dbscan')
        self.contamination = self.config_manager.get('anomaly_detection.methods.clustering.contamination', 0.05)
        
        self.logger.info(f"Clustering anomaly detector initialized with method: {self.method}, contamination: {self.contamination}")
    
    def detect_anomalies(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect anomalies using clustering methods.
        
        Args:
            df: DataFrame to analyze.
            table_name: Name of the table.
        
        Returns:
            Tuple of (anomalies DataFrame, normal DataFrame).
        """
        self.logger.info(f"Detecting clustering anomalies in table: {table_name}")
        
        # Create a copy of the DataFrame
        df_copy = df.copy()
        
        # Create a mask for anomalies
        anomaly_mask = pd.Series(False, index=df.index)
        
        # Select numeric columns
        numeric_df = df_copy.select_dtypes(include=np.number)
        
        if numeric_df.empty:
            self.logger.warning(f"No numeric columns found in table {table_name}")
            return pd.DataFrame(), df_copy
        
        # Fill missing values
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Apply clustering method
        if self.method == 'dbscan':
            anomaly_mask = self._detect_dbscan_anomalies(numeric_df)
        elif self.method == 'isolation_forest':
            anomaly_mask = self._detect_isolation_forest_anomalies(numeric_df)
        else:
            self.logger.warning(f"Unknown clustering method: {self.method}, falling back to DBSCAN")
            anomaly_mask = self._detect_dbscan_anomalies(numeric_df)
        
        # Split data into anomalies and normal
        anomalies_df = df_copy[anomaly_mask].copy()
        normal_df = df_copy[~anomaly_mask].copy()
        
        self.logger.info(f"Detected {len(anomalies_df)} clustering anomalies in table: {table_name}")
        return anomalies_df, normal_df
    
    def _detect_dbscan_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detect anomalies using DBSCAN clustering.
        
        Args:
            df: DataFrame to analyze.
        
        Returns:
            Boolean mask indicating anomalies.
        """
        self.logger.info("Detecting anomalies using DBSCAN clustering")
        
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df)
            
            # Apply DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(scaled_data)
            
            # Identify anomalies (points labeled as -1)
            anomaly_mask = pd.Series(clusters == -1, index=df.index)
            
            return anomaly_mask
        except ImportError:
            self.logger.warning("scikit-learn not installed, falling back to simple outlier detection")
            
            # Simple outlier detection
            anomaly_mask = pd.Series(False, index=df.index)
            
            for column in df.columns:
                mean = df[column].mean()
                std = df[column].std()
                
                # Identify outliers
                column_anomalies = np.abs(df[column] - mean) > 3 * std
                
                # Update mask
                anomaly_mask = anomaly_mask | column_anomalies
            
            return anomaly_mask
    
    def _detect_isolation_forest_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detect anomalies using Isolation Forest.
        
        Args:
            df: DataFrame to analyze.
        
        Returns:
            Boolean mask indicating anomalies.
        """
        self.logger.info("Detecting anomalies using Isolation Forest")
        
        try:
            from sklearn.ensemble import IsolationForest
            
            # Apply Isolation Forest
            isolation_forest = IsolationForest(contamination=self.contamination, random_state=42)
            predictions = isolation_forest.fit_predict(df)
            
            # Identify anomalies (points labeled as -1)
            anomaly_mask = pd.Series(predictions == -1, index=df.index)
            
            return anomaly_mask
        except ImportError:
            self.logger.warning("scikit-learn not installed, falling back to simple outlier detection")
            
            # Simple outlier detection
            anomaly_mask = pd.Series(False, index=df.index)
            
            for column in df.columns:
                mean = df[column].mean()
                std = df[column].std()
                
                # Identify outliers
                column_anomalies = np.abs(df[column] - mean) > 3 * std
                
                # Update mask
                anomaly_mask = anomaly_mask | column_anomalies
            
            return anomaly_mask


class AnomalyContextExtractor(Component):
    """Component for extracting context for anomalies."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the anomaly context extractor.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.relationships = None
    
    def initialize(self) -> None:
        """Initialize the anomaly context extractor.
        
        Raises:
            ConfigurationError: If the extractor cannot be initialized.
        """
        # Relationships will be loaded from data
        self.logger.info("Anomaly context extractor initialized")
    
    def extract_context(self, anomalies_df: pd.DataFrame, table_name: str, tables: Dict[str, pd.DataFrame], relationships: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Extract context for anomalies.
        
        This method identifies related records in other tables
        that provide context for the anomalies.
        
        Args:
            anomalies_df: DataFrame containing anomalies.
            table_name: Name of the table containing anomalies.
            tables: Dictionary mapping table names to DataFrames.
            relationships: List of relationship dictionaries.
        
        Returns:
            Dictionary mapping table names to DataFrames containing context records.
        """
        self.logger.info(f"Extracting context for anomalies in table: {table_name}")
        
        if anomalies_df.empty:
            self.logger.info(f"No anomalies found in table {table_name}, skipping context extraction")
            return {}
        
        # Store context records
        context = {}
        
        # Find relationships where this table is parent or child
        parent_relationships = [r for r in relationships if r.get('parent_table') == table_name]
        child_relationships = [r for r in relationships if r.get('child_table') == table_name]
        
        # Extract context from child tables
        for relationship in parent_relationships:
            child_table = relationship.get('child_table')
            parent_column = relationship.get('parent_column')
            child_column = relationship.get('child_column')
            
            if child_table not in tables:
                self.logger.warning(f"Child table {child_table} not found")
                continue
            
            if parent_column not in anomalies_df.columns:
                self.logger.warning(f"Parent column {parent_column} not found in table {table_name}")
                continue
            
            if child_column not in tables[child_table].columns:
                self.logger.warning(f"Child column {child_column} not found in table {child_table}")
                continue
            
            # Get values from anomalies
            parent_values = set(anomalies_df[parent_column].dropna())
            
            if not parent_values:
                continue
            
            # Find related records in child table
            child_df = tables[child_table]
            related_records = child_df[child_df[child_column].isin(parent_values)]
            
            if not related_records.empty:
                context[child_table] = related_records
        
        # Extract context from parent tables
        for relationship in child_relationships:
            parent_table = relationship.get('parent_table')
            parent_column = relationship.get('parent_column')
            child_column = relationship.get('child_column')
            
            if parent_table not in tables:
                self.logger.warning(f"Parent table {parent_table} not found")
                continue
            
            if child_column not in anomalies_df.columns:
                self.logger.warning(f"Child column {child_column} not found in table {table_name}")
                continue
            
            if parent_column not in tables[parent_table].columns:
                self.logger.warning(f"Parent column {parent_column} not found in table {parent_table}")
                continue
            
            # Get values from anomalies
            child_values = set(anomalies_df[child_column].dropna())
            
            if not child_values:
                continue
            
            # Find related records in parent table
            parent_df = tables[parent_table]
            related_records = parent_df[parent_df[parent_column].isin(child_values)]
            
            if not related_records.empty:
                context[parent_table] = related_records
        
        self.logger.info(f"Extracted context from {len(context)} related tables for anomalies in table: {table_name}")
        return context


class AnomalyDetectionPipeline(Pipeline):
    """Pipeline for anomaly detection and isolation."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the anomaly detection pipeline.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.statistical_detector = None
        self.rule_based_detector = None
        self.clustering_detector = None
        self.context_extractor = None
    
    def initialize(self) -> None:
        """Initialize the anomaly detection pipeline.
        
        Raises:
            ConfigurationError: If the pipeline cannot be initialized.
        """
        super().initialize()
        
        # Initialize components
        self.statistical_detector = StatisticalAnomalyDetector(self.config_manager)
        self.statistical_detector.initialize()
        
        self.rule_based_detector = RuleBasedAnomalyDetector(self.config_manager)
        self.rule_based_detector.initialize()
        
        self.clustering_detector = ClusteringAnomalyDetector(self.config_manager)
        self.clustering_detector.initialize()
        
        self.context_extractor = AnomalyContextExtractor(self.config_manager)
        self.context_extractor.initialize()
        
        # Initialize steps
        self.steps = [
            PipelineStep(self.detect_anomalies),
            PipelineStep(self.extract_context),
            PipelineStep(self.save_results)
        ]
        
        self.logger.info("Anomaly detection pipeline initialized")
    
    def detect_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in the data.
        
        Args:
            data: Input data with tables and domain partitions.
        
        Returns:
            Updated data with detected anomalies.
        
        Raises:
            ProcessingError: If anomaly detection fails.
        """
        self.logger.info("Detecting anomalies")
        
        try:
            tables = data.get('tables', {})
            
            if not tables:
                self.logger.warning("No tables found for anomaly detection")
                return data
            
            # Store anomalies and normal data
            anomalies = {}
            normal_data = {}
            
            # Process each table
            for table_name, df in tables.items():
                self.logger.info(f"Processing table: {table_name}")
                
                # Skip empty tables
                if df.empty:
                    self.logger.warning(f"Table {table_name} is empty, skipping")
                    normal_data[table_name] = df
                    continue
                
                # Apply statistical detector if enabled
                if self.config_manager.get('anomaly_detection.methods.statistical.enabled', True):
                    statistical_anomalies, df = self.statistical_detector.detect_anomalies(df, table_name)
                    
                    if not statistical_anomalies.empty:
                        if table_name not in anomalies:
                            anomalies[table_name] = statistical_anomalies
                        else:
                            anomalies[table_name] = pd.concat([anomalies[table_name], statistical_anomalies])
                
                # Apply rule-based detector if enabled
                if self.config_manager.get('anomaly_detection.methods.rule_based.enabled', True):
                    rule_based_anomalies, df = self.rule_based_detector.detect_anomalies(df, table_name)
                    
                    if not rule_based_anomalies.empty:
                        if table_name not in anomalies:
                            anomalies[table_name] = rule_based_anomalies
                        else:
                            anomalies[table_name] = pd.concat([anomalies[table_name], rule_based_anomalies])
                
                # Apply clustering detector if enabled
                if self.config_manager.get('anomaly_detection.methods.clustering.enabled', True):
                    clustering_anomalies, df = self.clustering_detector.detect_anomalies(df, table_name)
                    
                    if not clustering_anomalies.empty:
                        if table_name not in anomalies:
                            anomalies[table_name] = clustering_anomalies
                        else:
                            anomalies[table_name] = pd.concat([anomalies[table_name], clustering_anomalies])
                
                # Store normal data
                normal_data[table_name] = df
            
            # Remove duplicates in anomalies
            for table_name in anomalies:
                anomalies[table_name] = anomalies[table_name].drop_duplicates()
            
            # Update data
            result = data.copy()
            result['anomalies'] = anomalies
            result['normal_data'] = normal_data
            
            # Log summary
            total_anomalies = sum(len(df) for df in anomalies.values())
            self.logger.info(f"Detected {total_anomalies} anomalies across {len(anomalies)} tables")
            
            return result
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            raise ProcessingError(f"Error detecting anomalies: {str(e)}")
    
    def extract_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context for anomalies.
        
        Args:
            data: Input data with detected anomalies.
        
        Returns:
            Updated data with anomaly context.
        
        Raises:
            ProcessingError: If context extraction fails.
        """
        self.logger.info("Extracting context for anomalies")
        
        try:
            anomalies = data.get('anomalies', {})
            tables = data.get('tables', {})
            relationships = data.get('relationships', [])
            
            if not anomalies:
                self.logger.warning("No anomalies found for context extraction")
                return data
            
            if not relationships:
                self.logger.warning("No relationships found for context extraction")
                return data
            
            # Convert relationships to dictionaries if needed
            relationship_dicts = []
            for r in relationships:
                if hasattr(r, 'to_dict'):
                    relationship_dicts.append(r.to_dict())
                else:
                    relationship_dicts.append(r)
            
            # Store context
            context = {}
            
            # Process each table with anomalies
            for table_name, anomalies_df in anomalies.items():
                table_context = self.context_extractor.extract_context(anomalies_df, table_name, tables, relationship_dicts)
                
                if table_context:
                    context[table_name] = table_context
            
            # Update data
            result = data.copy()
            result['anomaly_context'] = context
            
            # Log summary
            total_context_tables = sum(len(ctx) for ctx in context.values())
            self.logger.info(f"Extracted context from {total_context_tables} related tables for anomalies")
            
            return result
        except Exception as e:
            self.logger.error(f"Error extracting context for anomalies: {str(e)}")
            raise ProcessingError(f"Error extracting context for anomalies: {str(e)}")
    
    def save_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save anomaly detection results.
        
        Args:
            data: Input data with detected anomalies and context.
        
        Returns:
            Updated data with saved results.
        
        Raises:
            ProcessingError: If saving results fails.
        """
        self.logger.info("Saving anomaly detection results")
        
        try:
            anomalies = data.get('anomalies', {})
            anomaly_context = data.get('anomaly_context', {})
            
            if not anomalies:
                self.logger.warning("No anomalies found to save")
                return data
            
            # Create output directories
            anomalies_dir = os.path.join(self.output_dir, 'anomalies')
            context_dir = os.path.join(self.output_dir, 'context')
            
            ensure_directory(anomalies_dir)
            ensure_directory(context_dir)
            
            # Save anomalies
            anomaly_files = {}
            for table_name, anomalies_df in anomalies.items():
                file_path = os.path.join(anomalies_dir, f"{table_name}_anomalies.csv")
                save_dataframe(anomalies_df, file_path)
                anomaly_files[table_name] = file_path
            
            # Save context
            context_files = {}
            for table_name, table_context in anomaly_context.items():
                table_context_dir = os.path.join(context_dir, table_name)
                ensure_directory(table_context_dir)
                
                table_context_files = {}
                for related_table, context_df in table_context.items():
                    file_path = os.path.join(table_context_dir, f"{related_table}_context.csv")
                    save_dataframe(context_df, file_path)
                    table_context_files[related_table] = file_path
                
                context_files[table_name] = table_context_files
            
            # Save summary
            summary = {
                'anomaly_counts': {table: len(df) for table, df in anomalies.items()},
                'context_counts': {table: {related: len(df) for related, df in table_context.items()} for table, table_context in anomaly_context.items()},
                'anomaly_files': anomaly_files,
                'context_files': context_files
            }
            
            summary_file = os.path.join(self.output_dir, 'anomaly_detection_summary.json')
            save_json(summary, summary_file)
            
            # Update data
            result = data.copy()
            result['anomaly_detection_results'] = {
                'anomaly_files': anomaly_files,
                'context_files': context_files,
                'summary_file': summary_file
            }
            
            self.logger.info("Anomaly detection results saved")
            return result
        except Exception as e:
            self.logger.error(f"Error saving anomaly detection results: {str(e)}")
            raise ProcessingError(f"Error saving anomaly detection results: {str(e)}")

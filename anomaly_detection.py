"""
Anomaly Detection and Isolation Module for the Data Warehouse Subsampling Framework.

This module implements the second layer of the data subsampling architecture,
responsible for identifying and extracting anomalies with contextual metadata
and storing them separately for preservation during the sampling process.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import pymongo

from ..common.base import Component, ConfigManager, PipelineStep, Pipeline, ProcessingError, ValidationError
from ..common.utils import validate_data_frame
from ..common.connectors import create_connector

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Representation of a data anomaly."""
    id: str
    table_name: str
    domain: str
    detection_method: str
    detection_score: float
    original_row: Dict[str, Any]
    context: Dict[str, Any]
    related_anomalies: List[str] = None
    created_at: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the anomaly to a dictionary.
        
        Returns:
            Dictionary representation of the anomaly.
        """
        return {
            'id': self.id,
            'table_name': self.table_name,
            'domain': self.domain,
            'detection_method': self.detection_method,
            'detection_score': self.detection_score,
            'original_row': self.original_row,
            'context': self.context,
            'related_anomalies': self.related_anomalies,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Anomaly':
        """Create an Anomaly from a dictionary.
        
        Args:
            data: Dictionary with anomaly information.
        
        Returns:
            Anomaly instance.
        """
        created_at = None
        if data.get('created_at'):
            try:
                created_at = datetime.fromisoformat(data['created_at'])
            except (ValueError, TypeError):
                created_at = None
        
        return cls(
            id=data.get('id', ''),
            table_name=data.get('table_name', ''),
            domain=data.get('domain', ''),
            detection_method=data.get('detection_method', ''),
            detection_score=data.get('detection_score', 0.0),
            original_row=data.get('original_row', {}),
            context=data.get('context', {}),
            related_anomalies=data.get('related_anomalies', []),
            created_at=created_at
        )


class AnomalyRepository(Component):
    """Repository for storing and retrieving anomalies."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the anomaly repository.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.repository_type = None
        self.connection = None
        self.file_path = None
        self.mongo_client = None
        self.mongo_db = None
        self.mongo_collection = None
    
    def initialize(self) -> None:
        """Initialize the anomaly repository.
        
        Raises:
            ConfigurationError: If the repository cannot be initialized.
        """
        # Get repository configuration
        repo_config = self.config_manager.get('anomaly_detection.anomaly_repository', {})
        self.repository_type = repo_config.get('type', 'file')
        
        if self.repository_type == 'file':
            # Create output directory
            output_dir = os.path.join(
                self.config_manager.get('general.output_directory', '/output/dwsf'),
                'anomaly_detection',
                'repository'
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Set file path
            self.file_path = os.path.join(output_dir, 'anomalies.json')
            
            self.logger.info(f"Anomaly repository initialized with file storage at {self.file_path}")
        
        elif self.repository_type == 'mongodb':
            # Get MongoDB connection parameters
            mongo_config = repo_config.get('connection', {})
            
            try:
                # Connect to MongoDB
                host = mongo_config.get('host', 'localhost')
                port = mongo_config.get('port', 27017)
                database = mongo_config.get('database', 'anomaly_repository')
                collection = mongo_config.get('collection', 'anomalies')
                user = mongo_config.get('user', '')
                password = mongo_config.get('password', '')
                
                # Build connection string
                if user and password:
                    connection_string = f"mongodb://{user}:{password}@{host}:{port}/{database}"
                else:
                    connection_string = f"mongodb://{host}:{port}/{database}"
                
                # Connect to MongoDB
                self.mongo_client = pymongo.MongoClient(connection_string)
                self.mongo_db = self.mongo_client[database]
                self.mongo_collection = self.mongo_db[collection]
                
                # Create indexes
                self.mongo_collection.create_index('id', unique=True)
                self.mongo_collection.create_index('table_name')
                self.mongo_collection.create_index('domain')
                self.mongo_collection.create_index('detection_method')
                
                self.logger.info(f"Anomaly repository initialized with MongoDB storage at {host}:{port}/{database}/{collection}")
            except Exception as e:
                self.logger.error(f"Error initializing MongoDB connection: {str(e)}")
                # Fall back to file storage
                self.repository_type = 'file'
                output_dir = os.path.join(
                    self.config_manager.get('general.output_directory', '/output/dwsf'),
                    'anomaly_detection',
                    'repository'
                )
                os.makedirs(output_dir, exist_ok=True)
                self.file_path = os.path.join(output_dir, 'anomalies.json')
                self.logger.info(f"Falling back to file storage at {self.file_path}")
        
        else:
            # Unsupported repository type, fall back to file
            self.repository_type = 'file'
            output_dir = os.path.join(
                self.config_manager.get('general.output_directory', '/output/dwsf'),
                'anomaly_detection',
                'repository'
            )
            os.makedirs(output_dir, exist_ok=True)
            self.file_path = os.path.join(output_dir, 'anomalies.json')
            self.logger.info(f"Unsupported repository type '{self.repository_type}', falling back to file storage at {self.file_path}")
    
    def validate(self) -> bool:
        """Validate the anomaly repository configuration and state.
        
        Returns:
            True if the repository is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if self.repository_type == 'file':
            # Check if the directory exists
            if not os.path.exists(os.path.dirname(self.file_path)):
                raise ValidationError(f"Repository directory does not exist: {os.path.dirname(self.file_path)}")
        
        elif self.repository_type == 'mongodb':
            # Check MongoDB connection
            if not self.mongo_client:
                raise ValidationError("MongoDB client is not initialized")
            
            try:
                # Test connection
                self.mongo_client.server_info()
            except Exception as e:
                raise ValidationError(f"MongoDB connection failed: {str(e)}")
        
        return True
    
    def save_anomalies(self, anomalies: List[Anomaly]) -> None:
        """Save anomalies to the repository.
        
        Args:
            anomalies: List of Anomaly instances to save.
        
        Raises:
            ProcessingError: If saving fails.
        """
        if not anomalies:
            self.logger.info("No anomalies to save")
            return
        
        self.logger.info(f"Saving {len(anomalies)} anomalies to repository")
        
        if self.repository_type == 'file':
            try:
                # Load existing anomalies
                existing_anomalies = []
                if os.path.exists(self.file_path):
                    with open(self.file_path, 'r') as f:
                        existing_data = json.load(f)
                        existing_anomalies = [Anomaly.from_dict(a) for a in existing_data]
                
                # Add new anomalies
                all_anomalies = existing_anomalies + anomalies
                
                # Remove duplicates based on ID
                unique_anomalies = {}
                for anomaly in all_anomalies:
                    unique_anomalies[anomaly.id] = anomaly
                
                # Convert to dictionaries
                anomaly_dicts = [anomaly.to_dict() for anomaly in unique_anomalies.values()]
                
                # Save to file
                with open(self.file_path, 'w') as f:
                    json.dump(anomaly_dicts, f, indent=2)
                
                self.logger.info(f"Saved {len(anomalies)} anomalies to file repository")
            except Exception as e:
                raise ProcessingError(f"Error saving anomalies to file: {str(e)}")
        
        elif self.repository_type == 'mongodb':
            try:
                # Convert anomalies to dictionaries
                anomaly_dicts = [anomaly.to_dict() for anomaly in anomalies]
                
                # Insert or update anomalies
                for anomaly_dict in anomaly_dicts:
                    self.mongo_collection.update_one(
                        {'id': anomaly_dict['id']},
                        {'$set': anomaly_dict},
                        upsert=True
                    )
                
                self.logger.info(f"Saved {len(anomalies)} anomalies to MongoDB repository")
            except Exception as e:
                raise ProcessingError(f"Error saving anomalies to MongoDB: {str(e)}")
    
    def get_anomalies(self, filters: Dict[str, Any] = None) -> List[Anomaly]:
        """Get anomalies from the repository.
        
        Args:
            filters: Dictionary of filters to apply.
        
        Returns:
            List of Anomaly instances.
        
        Raises:
            ProcessingError: If retrieval fails.
        """
        filters = filters or {}
        
        self.logger.info(f"Retrieving anomalies from repository with filters: {filters}")
        
        if self.repository_type == 'file':
            try:
                if not os.path.exists(self.file_path):
                    return []
                
                # Load anomalies from file
                with open(self.file_path, 'r') as f:
                    anomaly_dicts = json.load(f)
                
                # Convert to Anomaly instances
                anomalies = [Anomaly.from_dict(a) for a in anomaly_dicts]
                
                # Apply filters
                filtered_anomalies = []
                for anomaly in anomalies:
                    match = True
                    for key, value in filters.items():
                        if hasattr(anomaly, key):
                            if getattr(anomaly, key) != value:
                                match = False
                                break
                    
                    if match:
                        filtered_anomalies.append(anomaly)
                
                self.logger.info(f"Retrieved {len(filtered_anomalies)} anomalies from file repository")
                return filtered_anomalies
            except Exception as e:
                raise ProcessingError(f"Error retrieving anomalies from file: {str(e)}")
        
        elif self.repository_type == 'mongodb':
            try:
                # Query MongoDB
                cursor = self.mongo_collection.find(filters)
                
                # Convert to Anomaly instances
                anomalies = [Anomaly.from_dict(a) for a in cursor]
                
                self.logger.info(f"Retrieved {len(anomalies)} anomalies from MongoDB repository")
                return anomalies
            except Exception as e:
                raise ProcessingError(f"Error retrieving anomalies from MongoDB: {str(e)}")
        
        return []


class StatisticalAnomalyDetector(Component):
    """Component for detecting anomalies using statistical methods."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the statistical anomaly detector.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.z_score_threshold = None
        self.iqr_factor = None
    
    def initialize(self) -> None:
        """Initialize the statistical anomaly detector.
        
        Raises:
            ConfigurationError: If the detector cannot be initialized.
        """
        # Get configuration
        statistical_config = self.config_manager.get('anomaly_detection.methods.statistical', {})
        self.z_score_threshold = statistical_config.get('z_score_threshold', 3.0)
        self.iqr_factor = statistical_config.get('iqr_factor', 1.5)
        
        self.logger.info(f"Statistical anomaly detector initialized with z-score threshold {self.z_score_threshold} and IQR factor {self.iqr_factor}")
    
    def validate(self) -> bool:
        """Validate the statistical anomaly detector configuration and state.
        
        Returns:
            True if the detector is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if self.z_score_threshold <= 0:
            raise ValidationError(f"Invalid z-score threshold: {self.z_score_threshold}")
        
        if self.iqr_factor <= 0:
            raise ValidationError(f"Invalid IQR factor: {self.iqr_factor}")
        
        return True
    
    def detect_anomalies(self, df: pd.DataFrame, table_name: str, domain: str) -> Tuple[List[Anomaly], pd.DataFrame]:
        """Detect anomalies in a DataFrame using statistical methods.
        
        Args:
            df: DataFrame to analyze.
            table_name: Name of the table.
            domain: Domain name.
        
        Returns:
            Tuple of (list of detected anomalies, DataFrame with anomaly flags).
        """
        if not validate_data_frame(df):
            self.logger.warning(f"Invalid DataFrame for table {table_name}")
            return [], df
        
        self.logger.info(f"Detecting statistical anomalies in table {table_name}")
        
        # Create a copy of the DataFrame with anomaly flags
        result_df = df.copy()
        result_df['is_anomaly'] = False
        result_df['anomaly_score'] = 0.0
        result_df['anomaly_method'] = None
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_columns:
            self.logger.info(f"No numeric columns found in table {table_name}")
            return [], result_df
        
        # Detect anomalies using z-score
        z_score_anomalies = self._detect_z_score_anomalies(df, numeric_columns)
        
        # Detect anomalies using IQR
        iqr_anomalies = self._detect_iqr_anomalies(df, numeric_columns)
        
        # Combine anomalies
        anomaly_indices = z_score_anomalies.union(iqr_anomalies)
        
        if not anomaly_indices:
            self.logger.info(f"No statistical anomalies detected in table {table_name}")
            return [], result_df
        
        # Update result DataFrame
        for idx in anomaly_indices:
            if idx in z_score_anomalies:
                result_df.loc[idx, 'anomaly_method'] = 'z_score'
                result_df.loc[idx, 'anomaly_score'] = 1.0
            else:
                result_df.loc[idx, 'anomaly_method'] = 'iqr'
                result_df.loc[idx, 'anomaly_score'] = 0.9
            
            result_df.loc[idx, 'is_anomaly'] = True
        
        # Create Anomaly objects
        anomalies = []
        for idx in anomaly_indices:
            row = df.iloc[idx].to_dict()
            
            # Create context
            context = {
                'detection_columns': numeric_columns,
                'row_index': idx
            }
            
            # Create anomaly
            anomaly = Anomaly(
                id=f"{table_name}_{idx}",
                table_name=table_name,
                domain=domain,
                detection_method=result_df.loc[idx, 'anomaly_method'],
                detection_score=float(result_df.loc[idx, 'anomaly_score']),
                original_row=row,
                context=context,
                created_at=datetime.now()
            )
            
            anomalies.append(anomaly)
        
        self.logger.info(f"Detected {len(anomalies)} statistical anomalies in table {table_name}")
        return anomalies, result_df
    
    def _detect_z_score_anomalies(self, df: pd.DataFrame, numeric_columns: List[str]) -> set:
        """Detect anomalies using z-score method.
        
        Args:
            df: DataFrame to analyze.
            numeric_columns: List of numeric column names.
        
        Returns:
            Set of anomaly indices.
        """
        anomaly_indices = set()
        
        for column in numeric_columns:
            # Skip columns with all NaN values
            if df[column].isna().all():
                continue
            
            # Calculate z-scores
            mean = df[column].mean()
            std = df[column].std()
            
            if std == 0:
                continue
            
            z_scores = (df[column] - mean) / std
            
            # Find anomalies
            anomalies = df[abs(z_scores) > self.z_score_threshold].index
            anomaly_indices.update(anomalies)
        
        return anomaly_indices
    
    def _detect_iqr_anomalies(self, df: pd.DataFrame, numeric_columns: List[str]) -> set:
        """Detect anomalies using IQR method.
        
        Args:
            df: DataFrame to analyze.
            numeric_columns: List of numeric column names.
        
        Returns:
            Set of anomaly indices.
        """
        anomaly_indices = set()
        
        for column in numeric_columns:
            # Skip columns with all NaN values
            if df[column].isna().all():
                continue
            
            # Calculate IQR
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:
                continue
            
            # Define bounds
            lower_bound = q1 - (self.iqr_factor * iqr)
            upper_bound = q3 + (self.iqr_factor * iqr)
            
            # Find anomalies
            anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
            anomaly_indices.update(anomalies)
        
        return anomaly_indices


class PatternAnomalyDetector(Component):
    """Component for detecting anomalies using pattern recognition algorithms."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the pattern anomaly detector.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.algorithms = None
        self.models = {}
        self.output_dir = None
    
    def initialize(self) -> None:
        """Initialize the pattern anomaly detector.
        
        Raises:
            ConfigurationError: If the detector cannot be initialized.
        """
        # Get configuration
        pattern_config = self.config_manager.get('anomaly_detection.methods.pattern', {})
        self.algorithms = pattern_config.get('algorithms', ['isolation_forest', 'one_class_svm', 'local_outlier_factor'])
        
        # Create output directory for models
        self.output_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            'anomaly_detection',
            'models'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"Pattern anomaly detector initialized with algorithms: {self.algorithms}")
    
    def validate(self) -> bool:
        """Validate the pattern anomaly detector configuration and state.
        
        Returns:
            True if the detector is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if not self.algorithms:
            raise ValidationError("No pattern recognition algorithms specified")
        
        if not os.path.exists(self.output_dir):
            raise ValidationError(f"Model output directory does not exist: {self.output_dir}")
        
        return True
    
    def detect_anomalies(self, df: pd.DataFrame, table_name: str, domain: str) -> Tuple[List[Anomaly], pd.DataFrame]:
        """Detect anomalies in a DataFrame using pattern recognition algorithms.
        
        Args:
            df: DataFrame to analyze.
            table_name: Name of the table.
            domain: Domain name.
        
        Returns:
            Tuple of (list of detected anomalies, DataFrame with anomaly flags).
        """
        if not validate_data_frame(df):
            self.logger.warning(f"Invalid DataFrame for table {table_name}")
            return [], df
        
        self.logger.info(f"Detecting pattern anomalies in table {table_name}")
        
        # Create a copy of the DataFrame with anomaly flags
        result_df = df.copy()
        if 'is_anomaly' not in result_df.columns:
            result_df['is_anomaly'] = False
        if 'anomaly_score' not in result_df.columns:
            result_df['anomaly_score'] = 0.0
        if 'anomaly_method' not in result_df.columns:
            result_df['anomaly_method'] = None
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_columns:
            self.logger.info(f"No numeric columns found in table {table_name}")
            return [], result_df
        
        # Prepare data
        X = df[numeric_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Detect anomalies using each algorithm
        anomaly_indices = set()
        anomaly_scores = {}
        
        for algorithm in self.algorithms:
            try:
                if algorithm == 'isolation_forest':
                    anomalies, scores = self._detect_isolation_forest_anomalies(X_scaled, table_name)
                elif algorithm == 'one_class_svm':
                    anomalies, scores = self._detect_one_class_svm_anomalies(X_scaled, table_name)
                elif algorithm == 'local_outlier_factor':
                    anomalies, scores = self._detect_lof_anomalies(X_scaled, table_name)
                else:
                    self.logger.warning(f"Unsupported algorithm: {algorithm}")
                    continue
                
                # Update anomaly indices and scores
                anomaly_indices.update(anomalies)
                
                for idx, score in scores.items():
                    if idx not in anomaly_scores or score > anomaly_scores[idx][1]:
                        anomaly_scores[idx] = (algorithm, score)
            except Exception as e:
                self.logger.error(f"Error detecting anomalies with {algorithm}: {str(e)}")
        
        if not anomaly_indices:
            self.logger.info(f"No pattern anomalies detected in table {table_name}")
            return [], result_df
        
        # Update result DataFrame
        for idx in anomaly_indices:
            algorithm, score = anomaly_scores.get(idx, ('unknown', 0.8))
            
            # Only update if not already marked as anomaly or if score is higher
            if not result_df.loc[idx, 'is_anomaly'] or score > result_df.loc[idx, 'anomaly_score']:
                result_df.loc[idx, 'anomaly_method'] = algorithm
                result_df.loc[idx, 'anomaly_score'] = score
                result_df.loc[idx, 'is_anomaly'] = True
        
        # Create Anomaly objects
        anomalies = []
        for idx in anomaly_indices:
            # Skip if already marked as anomaly by another method with higher score
            if result_df.loc[idx, 'anomaly_method'] != anomaly_scores[idx][0]:
                continue
            
            row = df.iloc[idx].to_dict()
            
            # Create context
            context = {
                'detection_columns': numeric_columns,
                'row_index': idx,
                'algorithm': anomaly_scores[idx][0]
            }
            
            # Create anomaly
            anomaly = Anomaly(
                id=f"{table_name}_{idx}_{anomaly_scores[idx][0]}",
                table_name=table_name,
                domain=domain,
                detection_method=anomaly_scores[idx][0],
                detection_score=float(anomaly_scores[idx][1]),
                original_row=row,
                context=context,
                created_at=datetime.now()
            )
            
            anomalies.append(anomaly)
        
        self.logger.info(f"Detected {len(anomalies)} pattern anomalies in table {table_name}")
        return anomalies, result_df
    
    def _detect_isolation_forest_anomalies(self, X: np.ndarray, table_name: str) -> Tuple[set, Dict[int, float]]:
        """Detect anomalies using Isolation Forest algorithm.
        
        Args:
            X: Input data matrix.
            table_name: Name of the table.
        
        Returns:
            Tuple of (set of anomaly indices, dictionary mapping indices to anomaly scores).
        """
        # Check if model exists
        model_path = os.path.join(self.output_dir, f"{table_name}_isolation_forest.pkl")
        
        if os.path.exists(model_path):
            # Load existing model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            # Create and train new model
            model = IsolationForest(random_state=42, contamination=0.05)
            model.fit(X)
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Predict anomalies
        y_pred = model.predict(X)
        scores = model.decision_function(X)
        
        # Convert scores to anomaly scores (higher is more anomalous)
        anomaly_scores = 1 - (scores + 0.5)  # Scale to [0, 1]
        
        # Find anomaly indices
        anomaly_indices = set(np.where(y_pred == -1)[0])
        
        # Create score dictionary
        score_dict = {idx: anomaly_scores[idx] for idx in anomaly_indices}
        
        return anomaly_indices, score_dict
    
    def _detect_one_class_svm_anomalies(self, X: np.ndarray, table_name: str) -> Tuple[set, Dict[int, float]]:
        """Detect anomalies using One-Class SVM algorithm.
        
        Args:
            X: Input data matrix.
            table_name: Name of the table.
        
        Returns:
            Tuple of (set of anomaly indices, dictionary mapping indices to anomaly scores).
        """
        # Check if model exists
        model_path = os.path.join(self.output_dir, f"{table_name}_one_class_svm.pkl")
        
        if os.path.exists(model_path):
            # Load existing model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            # Create and train new model
            model = OneClassSVM(nu=0.05, kernel="rbf", gamma='scale')
            model.fit(X)
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Predict anomalies
        y_pred = model.predict(X)
        scores = model.decision_function(X)
        
        # Convert scores to anomaly scores (higher is more anomalous)
        anomaly_scores = -scores  # Negate scores so higher values are more anomalous
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())  # Scale to [0, 1]
        
        # Find anomaly indices
        anomaly_indices = set(np.where(y_pred == -1)[0])
        
        # Create score dictionary
        score_dict = {idx: anomaly_scores[idx] for idx in anomaly_indices}
        
        return anomaly_indices, score_dict
    
    def _detect_lof_anomalies(self, X: np.ndarray, table_name: str) -> Tuple[set, Dict[int, float]]:
        """Detect anomalies using Local Outlier Factor algorithm.
        
        Args:
            X: Input data matrix.
            table_name: Name of the table.
        
        Returns:
            Tuple of (set of anomaly indices, dictionary mapping indices to anomaly scores).
        """
        # Create model (LOF doesn't need to be saved as it's not a fitted model)
        model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        
        # Predict anomalies
        y_pred = model.fit_predict(X)
        scores = model.negative_outlier_factor_
        
        # Convert scores to anomaly scores (higher is more anomalous)
        anomaly_scores = -scores  # Negate scores so higher values are more anomalous
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())  # Scale to [0, 1]
        
        # Find anomaly indices
        anomaly_indices = set(np.where(y_pred == -1)[0])
        
        # Create score dictionary
        score_dict = {idx: anomaly_scores[idx] for idx in anomaly_indices}
        
        return anomaly_indices, score_dict


class BusinessRuleAnomalyDetector(Component):
    """Component for detecting anomalies using business rules."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the business rule anomaly detector.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.rules_file = None
        self.rules = {}
    
    def initialize(self) -> None:
        """Initialize the business rule anomaly detector.
        
        Raises:
            ConfigurationError: If the detector cannot be initialized.
        """
        # Get configuration
        business_rules_config = self.config_manager.get('anomaly_detection.methods.business_rules', {})
        self.rules_file = business_rules_config.get('rules_file')
        
        if self.rules_file:
            try:
                with open(self.rules_file, 'r') as f:
                    self.rules = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading business rules from {self.rules_file}: {str(e)}")
                self.rules = {}
        
        self.logger.info(f"Business rule anomaly detector initialized with {len(self.rules)} rules")
    
    def validate(self) -> bool:
        """Validate the business rule anomaly detector configuration and state.
        
        Returns:
            True if the detector is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        if not self.rules:
            self.logger.warning("No business rules defined")
            return True
        
        return True
    
    def detect_anomalies(self, df: pd.DataFrame, table_name: str, domain: str) -> Tuple[List[Anomaly], pd.DataFrame]:
        """Detect anomalies in a DataFrame using business rules.
        
        Args:
            df: DataFrame to analyze.
            table_name: Name of the table.
            domain: Domain name.
        
        Returns:
            Tuple of (list of detected anomalies, DataFrame with anomaly flags).
        """
        if not validate_data_frame(df):
            self.logger.warning(f"Invalid DataFrame for table {table_name}")
            return [], df
        
        self.logger.info(f"Detecting business rule anomalies in table {table_name}")
        
        # Create a copy of the DataFrame with anomaly flags
        result_df = df.copy()
        if 'is_anomaly' not in result_df.columns:
            result_df['is_anomaly'] = False
        if 'anomaly_score' not in result_df.columns:
            result_df['anomaly_score'] = 0.0
        if 'anomaly_method' not in result_df.columns:
            result_df['anomaly_method'] = None
        
        # Get rules for this table
        table_rules = self.rules.get(table_name, [])
        
        if not table_rules:
            self.logger.info(f"No business rules defined for table {table_name}")
            return [], result_df
        
        # Apply each rule
        anomaly_indices = set()
        rule_violations = {}
        
        for rule in table_rules:
            rule_name = rule.get('name', 'unnamed_rule')
            rule_condition = rule.get('condition')
            rule_score = rule.get('score', 0.8)
            
            if not rule_condition:
                continue
            
            try:
                # Apply rule condition
                mask = df.eval(rule_condition)
                
                # Find violations
                violations = df[mask].index
                
                # Update anomaly indices
                anomaly_indices.update(violations)
                
                # Store rule violations
                for idx in violations:
                    if idx not in rule_violations or rule_score > rule_violations[idx][1]:
                        rule_violations[idx] = (rule_name, rule_score)
            except Exception as e:
                self.logger.error(f"Error applying rule {rule_name}: {str(e)}")
        
        if not anomaly_indices:
            self.logger.info(f"No business rule anomalies detected in table {table_name}")
            return [], result_df
        
        # Update result DataFrame
        for idx in anomaly_indices:
            rule_name, score = rule_violations.get(idx, ('unknown_rule', 0.8))
            
            # Only update if not already marked as anomaly or if score is higher
            if not result_df.loc[idx, 'is_anomaly'] or score > result_df.loc[idx, 'anomaly_score']:
                result_df.loc[idx, 'anomaly_method'] = f"business_rule_{rule_name}"
                result_df.loc[idx, 'anomaly_score'] = score
                result_df.loc[idx, 'is_anomaly'] = True
        
        # Create Anomaly objects
        anomalies = []
        for idx in anomaly_indices:
            # Skip if already marked as anomaly by another method with higher score
            if not result_df.loc[idx, 'anomaly_method'].startswith('business_rule_'):
                continue
            
            row = df.iloc[idx].to_dict()
            
            # Create context
            context = {
                'rule_name': rule_violations[idx][0],
                'row_index': idx
            }
            
            # Create anomaly
            anomaly = Anomaly(
                id=f"{table_name}_{idx}_rule_{rule_violations[idx][0]}",
                table_name=table_name,
                domain=domain,
                detection_method=f"business_rule_{rule_violations[idx][0]}",
                detection_score=float(rule_violations[idx][1]),
                original_row=row,
                context=context,
                created_at=datetime.now()
            )
            
            anomalies.append(anomaly)
        
        self.logger.info(f"Detected {len(anomalies)} business rule anomalies in table {table_name}")
        return anomalies, result_df


class AnomalyDetectionPipeline(Pipeline):
    """Pipeline for anomaly detection and isolation."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the anomaly detection pipeline.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.statistical_detector = None
        self.pattern_detector = None
        self.business_rule_detector = None
        self.anomaly_repository = None
    
    def initialize(self) -> None:
        """Initialize the pipeline.
        
        Raises:
            ConfigurationError: If the pipeline cannot be initialized.
        """
        # Initialize components
        self.statistical_detector = StatisticalAnomalyDetector(self.config_manager)
        self.statistical_detector.initialize()
        
        self.pattern_detector = PatternAnomalyDetector(self.config_manager)
        self.pattern_detector.initialize()
        
        self.business_rule_detector = BusinessRuleAnomalyDetector(self.config_manager)
        self.business_rule_detector.initialize()
        
        self.anomaly_repository = AnomalyRepository(self.config_manager)
        self.anomaly_repository.initialize()
        
        # Add pipeline steps
        self.add_step(StatisticalAnomalyDetectionStep(self.config_manager, self.statistical_detector))
        self.add_step(PatternAnomalyDetectionStep(self.config_manager, self.pattern_detector))
        self.add_step(BusinessRuleAnomalyDetectionStep(self.config_manager, self.business_rule_detector))
        self.add_step(AnomalyStorageStep(self.config_manager, self.anomaly_repository))
        
        self.logger.info("Anomaly detection pipeline initialized")
    
    def validate(self) -> bool:
        """Validate the pipeline configuration and state.
        
        Returns:
            True if the pipeline is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        # Validate components
        self.statistical_detector.validate()
        self.pattern_detector.validate()
        self.business_rule_detector.validate()
        self.anomaly_repository.validate()
        
        return True


class StatisticalAnomalyDetectionStep(PipelineStep):
    """Pipeline step for detecting anomalies using statistical methods."""
    
    def __init__(self, config_manager: ConfigManager, detector: StatisticalAnomalyDetector):
        """Initialize the statistical anomaly detection step.
        
        Args:
            config_manager: Configuration manager instance.
            detector: StatisticalAnomalyDetector instance.
        """
        super().__init__(config_manager)
        self.detector = detector
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the statistical anomaly detection step.
        
        Args:
            input_data: Dictionary with data, relationships, and domain partitions.
        
        Returns:
            Dictionary with input data and detected anomalies.
        
        Raises:
            ProcessingError: If anomaly detection fails.
        """
        self.logger.info("Detecting statistical anomalies")
        
        if not input_data or 'domain_partitions' not in input_data:
            raise ProcessingError("No domain partitions to analyze")
        
        # Check if statistical anomaly detection is enabled
        if not self.config_manager.get('anomaly_detection.methods.statistical.enabled', True):
            self.logger.info("Statistical anomaly detection is disabled in configuration")
            return {**input_data, 'anomalies': []}
        
        # Detect anomalies in each domain partition
        all_anomalies = []
        domain_partitions = input_data['domain_partitions']
        
        for domain, tables in domain_partitions.items():
            for table_name, df in tables.items():
                try:
                    anomalies, df_with_flags = self.detector.detect_anomalies(df, table_name, domain)
                    all_anomalies.extend(anomalies)
                    
                    # Update DataFrame in domain partition
                    domain_partitions[domain][table_name] = df_with_flags
                except Exception as e:
                    self.logger.error(f"Error detecting statistical anomalies in table {table_name}: {str(e)}")
                    # Continue with other tables
        
        self.logger.info(f"Detected {len(all_anomalies)} statistical anomalies in total")
        
        return {
            **input_data,
            'domain_partitions': domain_partitions,
            'anomalies': all_anomalies
        }


class PatternAnomalyDetectionStep(PipelineStep):
    """Pipeline step for detecting anomalies using pattern recognition algorithms."""
    
    def __init__(self, config_manager: ConfigManager, detector: PatternAnomalyDetector):
        """Initialize the pattern anomaly detection step.
        
        Args:
            config_manager: Configuration manager instance.
            detector: PatternAnomalyDetector instance.
        """
        super().__init__(config_manager)
        self.detector = detector
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pattern anomaly detection step.
        
        Args:
            input_data: Dictionary with data, relationships, domain partitions, and anomalies.
        
        Returns:
            Dictionary with input data and updated anomalies.
        
        Raises:
            ProcessingError: If anomaly detection fails.
        """
        self.logger.info("Detecting pattern anomalies")
        
        if not input_data or 'domain_partitions' not in input_data:
            raise ProcessingError("No domain partitions to analyze")
        
        # Check if pattern anomaly detection is enabled
        if not self.config_manager.get('anomaly_detection.methods.pattern.enabled', True):
            self.logger.info("Pattern anomaly detection is disabled in configuration")
            return input_data
        
        # Get existing anomalies
        existing_anomalies = input_data.get('anomalies', [])
        
        # Detect anomalies in each domain partition
        new_anomalies = []
        domain_partitions = input_data['domain_partitions']
        
        for domain, tables in domain_partitions.items():
            for table_name, df in tables.items():
                try:
                    anomalies, df_with_flags = self.detector.detect_anomalies(df, table_name, domain)
                    new_anomalies.extend(anomalies)
                    
                    # Update DataFrame in domain partition
                    domain_partitions[domain][table_name] = df_with_flags
                except Exception as e:
                    self.logger.error(f"Error detecting pattern anomalies in table {table_name}: {str(e)}")
                    # Continue with other tables
        
        self.logger.info(f"Detected {len(new_anomalies)} pattern anomalies in total")
        
        # Combine anomalies
        all_anomalies = existing_anomalies + new_anomalies
        
        return {
            **input_data,
            'domain_partitions': domain_partitions,
            'anomalies': all_anomalies
        }


class BusinessRuleAnomalyDetectionStep(PipelineStep):
    """Pipeline step for detecting anomalies using business rules."""
    
    def __init__(self, config_manager: ConfigManager, detector: BusinessRuleAnomalyDetector):
        """Initialize the business rule anomaly detection step.
        
        Args:
            config_manager: Configuration manager instance.
            detector: BusinessRuleAnomalyDetector instance.
        """
        super().__init__(config_manager)
        self.detector = detector
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the business rule anomaly detection step.
        
        Args:
            input_data: Dictionary with data, relationships, domain partitions, and anomalies.
        
        Returns:
            Dictionary with input data and updated anomalies.
        
        Raises:
            ProcessingError: If anomaly detection fails.
        """
        self.logger.info("Detecting business rule anomalies")
        
        if not input_data or 'domain_partitions' not in input_data:
            raise ProcessingError("No domain partitions to analyze")
        
        # Check if business rule anomaly detection is enabled
        if not self.config_manager.get('anomaly_detection.methods.business_rules.enabled', True):
            self.logger.info("Business rule anomaly detection is disabled in configuration")
            return input_data
        
        # Get existing anomalies
        existing_anomalies = input_data.get('anomalies', [])
        
        # Detect anomalies in each domain partition
        new_anomalies = []
        domain_partitions = input_data['domain_partitions']
        
        for domain, tables in domain_partitions.items():
            for table_name, df in tables.items():
                try:
                    anomalies, df_with_flags = self.detector.detect_anomalies(df, table_name, domain)
                    new_anomalies.extend(anomalies)
                    
                    # Update DataFrame in domain partition
                    domain_partitions[domain][table_name] = df_with_flags
                except Exception as e:
                    self.logger.error(f"Error detecting business rule anomalies in table {table_name}: {str(e)}")
                    # Continue with other tables
        
        self.logger.info(f"Detected {len(new_anomalies)} business rule anomalies in total")
        
        # Combine anomalies
        all_anomalies = existing_anomalies + new_anomalies
        
        return {
            **input_data,
            'domain_partitions': domain_partitions,
            'anomalies': all_anomalies
        }


class AnomalyStorageStep(PipelineStep):
    """Pipeline step for storing detected anomalies."""
    
    def __init__(self, config_manager: ConfigManager, repository: AnomalyRepository):
        """Initialize the anomaly storage step.
        
        Args:
            config_manager: Configuration manager instance.
            repository: AnomalyRepository instance.
        """
        super().__init__(config_manager)
        self.repository = repository
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the anomaly storage step.
        
        Args:
            input_data: Dictionary with data, relationships, domain partitions, and anomalies.
        
        Returns:
            Dictionary with input data and stored anomalies.
        
        Raises:
            ProcessingError: If anomaly storage fails.
        """
        self.logger.info("Storing detected anomalies")
        
        if not input_data:
            raise ProcessingError("No input data")
        
        # Get anomalies
        anomalies = input_data.get('anomalies', [])
        
        if not anomalies:
            self.logger.info("No anomalies to store")
            return input_data
        
        # Store anomalies
        try:
            self.repository.save_anomalies(anomalies)
            self.logger.info(f"Stored {len(anomalies)} anomalies in repository")
        except Exception as e:
            self.logger.error(f"Error storing anomalies: {str(e)}")
            # Continue with pipeline
        
        return input_data

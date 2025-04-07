"""
Base classes and interfaces for the Data Warehouse Subsampling Framework.

This module provides the core abstractions that are used throughout the framework,
including base classes for components, interfaces for extensibility, and common
utilities for logging, configuration, and error handling.
"""

import abc
import logging
import os
import yaml
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""
    pass


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


class ProcessingError(Exception):
    """Exception raised for errors during data processing."""
    pass


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    type: str
    connection_params: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataSourceConfig':
        """Create a DataSourceConfig from a dictionary."""
        return cls(
            type=config_dict.get('type', ''),
            connection_params={k: v for k, v in config_dict.items() if k != 'type'}
        )


class ConfigManager:
    """Manages configuration for the framework."""
    
    def __init__(self, config_path: str):
        """Initialize the ConfigManager.
        
        Args:
            config_path: Path to the configuration file.
        
        Raises:
            ConfigurationError: If the configuration file cannot be loaded.
        """
        self.config_path = config_path
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from the config file.
        
        Raises:
            ConfigurationError: If the configuration file cannot be loaded.
        """
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Set log level from config
            log_level = self.get('general.log_level', 'INFO')
            logging.getLogger().setLevel(getattr(logging, log_level))
            
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get a configuration value by path.
        
        Args:
            path: Dot-separated path to the configuration value.
            default: Default value to return if the path is not found.
        
        Returns:
            The configuration value, or the default if not found.
        """
        parts = path.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def get_data_source_config(self, source_name: str) -> Optional[DataSourceConfig]:
        """Get configuration for a data source.
        
        Args:
            source_name: Name of the data source.
        
        Returns:
            DataSourceConfig for the data source, or None if not found.
        """
        source_config = self.get(f'data_sources.{source_name}')
        if source_config:
            return DataSourceConfig.from_dict(source_config)
        return None


class Component(abc.ABC):
    """Base class for all framework components."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the component.
        
        Args:
            config_manager: Configuration manager instance.
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the component.
        
        This method should be called before using the component.
        
        Raises:
            ConfigurationError: If the component cannot be initialized due to configuration issues.
        """
        pass
    
    @abc.abstractmethod
    def validate(self) -> bool:
        """Validate the component configuration and state.
        
        Returns:
            True if the component is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        pass


class DataConnector(Component):
    """Base class for data source connectors."""
    
    def __init__(self, config_manager: ConfigManager, source_name: str):
        """Initialize the data connector.
        
        Args:
            config_manager: Configuration manager instance.
            source_name: Name of the data source.
        
        Raises:
            ConfigurationError: If the data source configuration is invalid.
        """
        super().__init__(config_manager)
        self.source_name = source_name
        self.source_config = config_manager.get_data_source_config(source_name)
        
        if not self.source_config:
            raise ConfigurationError(f"Data source '{source_name}' not found in configuration")
    
    @abc.abstractmethod
    def connect(self) -> Any:
        """Connect to the data source.
        
        Returns:
            Connection object or client.
        
        Raises:
            ConnectionError: If connection fails.
        """
        pass
    
    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the data source."""
        pass
    
    @abc.abstractmethod
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table.
        
        Args:
            table_name: Name of the table.
        
        Returns:
            Dictionary with schema information.
        """
        pass
    
    @abc.abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query on the data source.
        
        Args:
            query: Query string.
            params: Query parameters.
        
        Returns:
            Query result.
        """
        pass


class Pipeline(Component):
    """Base class for data processing pipelines."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the pipeline.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.steps = []
    
    def add_step(self, step: 'PipelineStep') -> None:
        """Add a step to the pipeline.
        
        Args:
            step: Pipeline step to add.
        """
        self.steps.append(step)
    
    def run(self) -> Any:
        """Run the pipeline.
        
        Returns:
            Result of the pipeline execution.
        
        Raises:
            ProcessingError: If pipeline execution fails.
        """
        result = None
        for step in self.steps:
            try:
                self.logger.info(f"Running pipeline step: {step.__class__.__name__}")
                result = step.execute(result)
            except Exception as e:
                self.logger.error(f"Pipeline step {step.__class__.__name__} failed: {str(e)}")
                raise ProcessingError(f"Pipeline execution failed at step {step.__class__.__name__}: {str(e)}")
        
        return result


class PipelineStep(abc.ABC):
    """Base class for pipeline steps."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the pipeline step.
        
        Args:
            config_manager: Configuration manager instance.
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abc.abstractmethod
    def execute(self, input_data: Any) -> Any:
        """Execute the pipeline step.
        
        Args:
            input_data: Input data from the previous step.
        
        Returns:
            Result of the step execution.
        
        Raises:
            ProcessingError: If step execution fails.
        """
        pass


class DataProcessor(Component):
    """Base class for data processors."""
    
    @abc.abstractmethod
    def process(self, data: Any) -> Any:
        """Process data.
        
        Args:
            data: Input data.
        
        Returns:
            Processed data.
        
        Raises:
            ProcessingError: If processing fails.
        """
        pass


class AnomalyDetector(DataProcessor):
    """Base class for anomaly detectors."""
    
    @abc.abstractmethod
    def detect(self, data: Any) -> Dict[str, Any]:
        """Detect anomalies in data.
        
        Args:
            data: Input data.
        
        Returns:
            Dictionary with anomaly detection results.
        
        Raises:
            ProcessingError: If anomaly detection fails.
        """
        pass


class SamplingTechnique(DataProcessor):
    """Base class for sampling techniques."""
    
    @abc.abstractmethod
    def sample(self, data: Any, sample_size: float) -> Any:
        """Sample data.
        
        Args:
            data: Input data.
            sample_size: Sample size as a fraction of the input data.
        
        Returns:
            Sampled data.
        
        Raises:
            ProcessingError: If sampling fails.
        """
        pass


class DataClassifier(DataProcessor):
    """Base class for data classifiers."""
    
    @abc.abstractmethod
    def classify(self, data: Any) -> Dict[str, Any]:
        """Classify data.
        
        Args:
            data: Input data.
        
        Returns:
            Dictionary with classification results.
        
        Raises:
            ProcessingError: If classification fails.
        """
        pass


class DataIntegrator(DataProcessor):
    """Base class for data integrators."""
    
    @abc.abstractmethod
    def integrate(self, normal_data: Any, anomaly_data: Any) -> Any:
        """Integrate normal data and anomaly data.
        
        Args:
            normal_data: Normal data.
            anomaly_data: Anomaly data.
        
        Returns:
            Integrated data.
        
        Raises:
            ProcessingError: If integration fails.
        """
        pass


class EnvironmentProvisioner(Component):
    """Base class for environment provisioners."""
    
    @abc.abstractmethod
    def provision(self, dataset: Any, env_name: str) -> Dict[str, Any]:
        """Provision a test environment.
        
        Args:
            dataset: Dataset to provision.
            env_name: Name of the environment.
        
        Returns:
            Dictionary with environment details.
        
        Raises:
            ProcessingError: If provisioning fails.
        """
        pass


class DataWarehouseSubsamplingFramework:
    """Main framework class that orchestrates all components."""
    
    def __init__(self, config_path: str):
        """Initialize the framework.
        
        Args:
            config_path: Path to the configuration file.
        
        Raises:
            ConfigurationError: If the configuration file cannot be loaded.
        """
        self.config_manager = ConfigManager(config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.data_classification = None
        self.anomaly_detection = None
        self.core_sampling = None
        self.data_integration = None
        self.test_env_provisioning = None
    
    def initialize(self) -> None:
        """Initialize all framework components.
        
        Raises:
            ConfigurationError: If components cannot be initialized.
        """
        self.logger.info("Initializing Data Warehouse Subsampling Framework")
        
        # Create output directory if it doesn't exist
        output_dir = self.config_manager.get('general.output_directory', '/output/dwsf')
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components based on configuration
        # These will be implemented in their respective modules
        pass
    
    def run(self) -> None:
        """Run the complete framework pipeline.
        
        Raises:
            ProcessingError: If pipeline execution fails.
        """
        self.logger.info("Running Data Warehouse Subsampling Framework")
        
        # Run each stage of the pipeline
        self.run_data_classification()
        self.run_anomaly_detection()
        self.run_core_sampling()
        self.run_data_integration()
        self.run_test_env_provisioning()
        
        self.logger.info("Data Warehouse Subsampling Framework execution completed successfully")
    
    def run_data_classification(self) -> None:
        """Run the data classification stage."""
        if self.config_manager.get('data_classification.enabled', True):
            self.logger.info("Running data classification stage")
            # Implementation will be added in the data_classification module
        else:
            self.logger.info("Data classification stage is disabled in configuration")
    
    def run_anomaly_detection(self) -> None:
        """Run the anomaly detection stage."""
        if self.config_manager.get('anomaly_detection.enabled', True):
            self.logger.info("Running anomaly detection stage")
            # Implementation will be added in the anomaly_detection module
        else:
            self.logger.info("Anomaly detection stage is disabled in configuration")
    
    def run_core_sampling(self) -> None:
        """Run the core sampling stage."""
        if self.config_manager.get('core_sampling.enabled', True):
            self.logger.info("Running core sampling stage")
            # Implementation will be added in the core_sampling module
        else:
            self.logger.info("Core sampling stage is disabled in configuration")
    
    def run_data_integration(self) -> None:
        """Run the data integration stage."""
        if self.config_manager.get('data_integration.enabled', True):
            self.logger.info("Running data integration stage")
            # Implementation will be added in the data_integration module
        else:
            self.logger.info("Data integration stage is disabled in configuration")
    
    def run_test_env_provisioning(self) -> None:
        """Run the test environment provisioning stage."""
        if self.config_manager.get('test_env_provisioning.enabled', True):
            self.logger.info("Running test environment provisioning stage")
            # Implementation will be added in the test_env_provisioning module
        else:
            self.logger.info("Test environment provisioning stage is disabled in configuration")

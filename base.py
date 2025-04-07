"""
Base classes for the Data Warehouse Subsampling Framework.

This module provides the base classes used throughout the framework,
including Component, Pipeline, and ConfigManager.
"""

import os
import logging
import yaml
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dwsf.log')
    ]
)


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


class ProcessingError(Exception):
    """Exception raised for processing errors."""
    pass


class ConfigManager:
    """Configuration manager for the framework."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the configuration manager.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get a configuration value by path.
        
        Args:
            path: Dot-separated path to the configuration value.
            default: Default value to return if the path is not found.
        
        Returns:
            The configuration value at the specified path, or the default value.
        """
        parts = path.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set(self, path: str, value: Any) -> None:
        """Set a configuration value by path.
        
        Args:
            path: Dot-separated path to the configuration value.
            value: Value to set.
        """
        parts = path.split('.')
        config = self.config
        
        for i, part in enumerate(parts[:-1]):
            if part not in config:
                config[part] = {}
            config = config[part]
        
        config[parts[-1]] = value
    
    @classmethod
    def from_file(cls, file_path: str) -> 'ConfigManager':
        """Create a configuration manager from a YAML file.
        
        Args:
            file_path: Path to the YAML configuration file.
        
        Returns:
            A new ConfigManager instance.
        
        Raises:
            ConfigurationError: If the file cannot be read or parsed.
        """
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            
            return cls(config or {})
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {file_path}: {str(e)}")


class Component:
    """Base class for all components in the framework."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the component.
        
        Args:
            config_manager: Configuration manager instance.
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def initialize(self) -> None:
        """Initialize the component.
        
        This method should be overridden by subclasses to perform
        any necessary initialization.
        
        Raises:
            ConfigurationError: If the component cannot be initialized.
        """
        pass
    
    def validate(self) -> bool:
        """Validate the component configuration and state.
        
        This method should be overridden by subclasses to perform
        any necessary validation.
        
        Returns:
            True if the component is valid, False otherwise.
        
        Raises:
            ValidationError: If validation fails.
        """
        return True


@dataclass
class PipelineStep:
    """A step in a pipeline."""
    
    function: Callable
    name: Optional[str] = None
    enabled: bool = True
    
    def __post_init__(self):
        """Initialize the step name if not provided."""
        if self.name is None:
            self.name = self.function.__name__


class Pipeline(Component):
    """Base class for all pipelines in the framework."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the pipeline.
        
        Args:
            config_manager: Configuration manager instance.
        """
        super().__init__(config_manager)
        self.steps: List[PipelineStep] = []
        self.output_dir = None
    
    def initialize(self) -> None:
        """Initialize the pipeline.
        
        This method should be overridden by subclasses to set up
        the pipeline steps and perform any necessary initialization.
        
        Raises:
            ConfigurationError: If the pipeline cannot be initialized.
        """
        # Create output directory
        self.output_dir = os.path.join(
            self.config_manager.get('general.output_directory', '/output/dwsf'),
            self.__class__.__name__.lower().replace('pipeline', '')
        )
        os.makedirs(self.output_dir, exist_ok=True)
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline.
        
        This method executes each step in the pipeline in order.
        
        Args:
            data: Input data for the pipeline.
        
        Returns:
            The output data from the pipeline.
        
        Raises:
            ProcessingError: If pipeline execution fails.
        """
        self.logger.info(f"Executing pipeline: {self.__class__.__name__}")
        
        result = data.copy()
        
        for step in self.steps:
            if step.enabled:
                self.logger.info(f"Executing step: {step.name}")
                try:
                    result = step.function(result)
                except Exception as e:
                    self.logger.error(f"Error in step {step.name}: {str(e)}")
                    raise ProcessingError(f"Error in step {step.name}: {str(e)}")
        
        self.logger.info(f"Pipeline completed: {self.__class__.__name__}")
        return result


@dataclass
class Relationship:
    """A relationship between two tables."""
    
    parent_table: str
    parent_column: str
    child_table: str
    child_column: str
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the relationship to a dictionary.
        
        Returns:
            A dictionary representation of the relationship.
        """
        return {
            'parent_table': self.parent_table,
            'parent_column': self.parent_column,
            'child_table': self.child_table,
            'child_column': self.child_column,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Create a relationship from a dictionary.
        
        Args:
            data: Dictionary representation of the relationship.
        
        Returns:
            A new Relationship instance.
        """
        return cls(
            parent_table=data['parent_table'],
            parent_column=data['parent_column'],
            child_table=data['child_table'],
            child_column=data['child_column'],
            confidence=data.get('confidence', 1.0)
        )


@dataclass
class SamplingResult:
    """Result of a sampling operation."""
    
    table_name: str
    domain: str
    original_row_count: int
    sampled_row_count: int
    sampling_method: str
    sampling_rate: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the sampling result to a dictionary.
        
        Returns:
            A dictionary representation of the sampling result.
        """
        return {
            'table_name': self.table_name,
            'domain': self.domain,
            'original_row_count': self.original_row_count,
            'sampled_row_count': self.sampled_row_count,
            'sampling_method': self.sampling_method,
            'sampling_rate': self.sampling_rate,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SamplingResult':
        """Create a sampling result from a dictionary.
        
        Args:
            data: Dictionary representation of the sampling result.
        
        Returns:
            A new SamplingResult instance.
        """
        return cls(
            table_name=data['table_name'],
            domain=data['domain'],
            original_row_count=data['original_row_count'],
            sampled_row_count=data['sampled_row_count'],
            sampling_method=data['sampling_method'],
            sampling_rate=data['sampling_rate'],
            metadata=data.get('metadata', {})
        )


@dataclass
class IntegrationResult:
    """Result of a data integration operation."""
    
    name: str
    tables: List[str]
    row_counts: Dict[str, int]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the integration result to a dictionary.
        
        Returns:
            A dictionary representation of the integration result.
        """
        return {
            'name': self.name,
            'tables': self.tables,
            'row_counts': self.row_counts,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntegrationResult':
        """Create an integration result from a dictionary.
        
        Args:
            data: Dictionary representation of the integration result.
        
        Returns:
            A new IntegrationResult instance.
        """
        return cls(
            name=data['name'],
            tables=data['tables'],
            row_counts=data['row_counts'],
            metadata=data.get('metadata', {})
        )


@dataclass
class EnvironmentStatus:
    """Status of a test environment."""
    
    name: str
    status: str  # 'running', 'stopped', 'failed'
    connection_info: Dict[str, Any] = None
    error: str = None
    
    def __post_init__(self):
        """Initialize connection_info if not provided."""
        if self.connection_info is None:
            self.connection_info = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the environment status to a dictionary.
        
        Returns:
            A dictionary representation of the environment status.
        """
        return {
            'name': self.name,
            'status': self.status,
            'connection_info': self.connection_info,
            'error': self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentStatus':
        """Create an environment status from a dictionary.
        
        Args:
            data: Dictionary representation of the environment status.
        
        Returns:
            A new EnvironmentStatus instance.
        """
        return cls(
            name=data['name'],
            status=data['status'],
            connection_info=data.get('connection_info', {}),
            error=data.get('error')
        )

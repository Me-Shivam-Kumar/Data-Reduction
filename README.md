# Data Warehouse Subsampling Framework

A comprehensive solution for reducing data warehouse testing volumes while preserving anomalies and data relationships.

## Overview

The Data Warehouse Subsampling Framework (DWSF) addresses the challenge of managing large data volumes (35TB+) in data warehouse testing environments. By implementing intelligent data reduction techniques while preserving anomalies and data relationships, this framework enables:

- 80-90% reduction in test data volume
- 60-80% reduction in test execution time
- Preservation of all anomalies with their contextual data
- Maintenance of referential integrity across datasets
- On-demand test environment provisioning

## Architecture

The framework implements a five-layer architecture:

1. **Data Classification & Partitioning Layer**: Analyzes and classifies data across business domains, identifies data relationships and dependencies, and partitions data for optimal parallel processing.

2. **Anomaly Detection & Isolation Layer**: Employs multiple detection algorithms in parallel, identifies and extracts anomalies across all data domains, and stores anomalies separately with contextual metadata.

3. **Core Sampling Layer**: Applies domain-specific sampling techniques in parallel, uses stratified sampling for transactional data, implements entity-based subsetting for master data, and extracts boundary values for reference data.

4. **Data Integration Layer**: Maintains referential integrity across sampled datasets, ensures anomaly-to-normal data relationships are preserved, and creates purpose-specific integrated test datasets.

5. **Test Environment Provisioning Layer**: Uses virtualization to minimize storage requirements, implements copy-on-write for efficient environment creation, and provides on-demand test environment provisioning.

## Key Features

- **Intelligent Data Reduction**: Multiple sampling techniques (stratified, entity-based, boundary value) applied based on data characteristics
- **Anomaly Preservation**: Complete preservation of all anomalies with their contextual data
- **Referential Integrity**: Maintenance of relationships between entities across sampled datasets
- **Parallel Processing**: Optimized performance through parallel execution of independent operations
- **Flexible Environment Provisioning**: Support for Docker, file-based, and database-based test environments
- **Comprehensive Reporting**: Detailed metrics on data reduction, anomaly preservation, and environment status

## Installation

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized test environments)
- Pandas, NumPy, and other dependencies listed in requirements.txt

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-organization/data-warehouse-subsampling-framework.git
   cd data-warehouse-subsampling-framework
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the framework by editing `config.yaml` to match your environment.

## Usage

### Basic Usage

Run the framework with default settings:

```
python -m dwsf.main --config config.yaml
```

### Advanced Options

- Enable parallel processing:
  ```
  python -m dwsf.main --config config.yaml --parallel
  ```

- Specify custom output directory:
  ```
  python -m dwsf.main --config config.yaml --output-dir /path/to/output
  ```

### Configuration

The framework is configured through a YAML file. Key configuration sections include:

- **General**: Basic framework settings
- **Data Classification**: Settings for data domain partitioning
- **Anomaly Detection**: Algorithm selection and parameters
- **Core Sampling**: Sampling method configurations
- **Data Integration**: Referential integrity and purpose-specific dataset settings
- **Test Environment Provisioning**: Environment type and parameters
- **Orchestration**: Workflow execution settings

See `config.example.yaml` for a complete example with documentation.

## Code Structure

```
dwsf/
├── common/                  # Common utilities and base classes
│   ├── base.py              # Base component and pipeline classes
│   ├── utils.py             # Utility functions
│   └── connectors.py        # Data source connectors
├── data_classification/     # Data classification and partitioning
│   └── data_classification.py
├── anomaly_detection/       # Anomaly detection and isolation
│   └── anomaly_detection.py
├── core_sampling/           # Core sampling techniques
│   └── core_sampling.py
├── data_integration/        # Data integration and referential integrity
│   └── data_integration.py
├── test_env_provisioning/   # Test environment provisioning
│   └── test_env_provisioning.py
├── orchestration/           # Workflow orchestration
│   └── orchestration.py
└── main.py                  # Main entry point
```

## Implementation Details

### Data Classification & Partitioning

The data classification module analyzes the structure of input data, identifies relationships between tables, and partitions data into logical domains. This enables domain-specific processing in subsequent steps.

Key components:
- Domain identification
- Relationship discovery
- Data partitioning

### Anomaly Detection & Isolation

The anomaly detection module employs multiple algorithms to identify anomalies in the data. Anomalies are isolated and stored separately with contextual metadata.

Supported algorithms:
- Statistical methods (Z-score, IQR)
- Clustering-based detection
- Isolation Forest
- Custom rule-based detection

### Core Sampling

The core sampling module applies different sampling techniques based on data characteristics:

- **Stratified Sampling**: Preserves distribution of key attributes
- **Entity-Based Subsetting**: Maintains complete entities with related records
- **Boundary Value Sampling**: Preserves edge cases and critical values
- **Anomaly Preservation Sampling**: Ensures all anomalies are included

### Data Integration

The data integration module ensures that sampled datasets maintain referential integrity and that anomalies are preserved with their contextual data.

Key features:
- Referential integrity maintenance
- Anomaly context integration
- Purpose-specific dataset creation
- Dataset export in multiple formats

### Test Environment Provisioning

The test environment provisioning module creates test environments based on the integrated datasets.

Supported environment types:
- Docker containers (PostgreSQL, MySQL)
- File-based (CSV, Parquet, JSON)
- Database (SQLite)

### Orchestration

The orchestration module ties everything together, providing both sequential and parallel workflow execution options.

Features:
- Workflow management
- Parallel processing
- Comprehensive logging
- Detailed reporting

## Performance Considerations

- For datasets under 1TB, all operations can typically run on a single machine
- For larger datasets (1-10TB), distributed processing is recommended
- For very large datasets (10TB+), consider running the framework in a cloud environment with adequate resources

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

"""
Setup script for the Data Warehouse Subsampling Framework.

This module provides the setup configuration for installing the framework
as a Python package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dwsf",
    version="1.0.0",
    author="DWSF Team",
    author_email="dwsf@example.com",
    description="A comprehensive solution for reducing data warehouse testing volumes while preserving anomalies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-organization/data-warehouse-subsampling-framework",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Database",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
        "docker>=5.0.0",
        "pyyaml>=6.0",
        "matplotlib>=3.4.0",
        "sqlalchemy>=1.4.0",
        "pyarrow>=6.0.0",
        "fastparquet>=0.8.0",
        "psycopg2-binary>=2.9.0",
        "pymysql>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
            "mypy>=0.812",
        ],
    },
    entry_points={
        "console_scripts": [
            "dwsf=dwsf.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dwsf": ["config.example.yaml"],
    },
)

#!/usr/bin/env python3
"""
Setup script for the Data Warehouse Subsampling Framework.
"""

from setuptools import setup, find_packages

setup(
    name="dwsf",
    version="1.0.0",
    description="Data Warehouse Subsampling Framework",
    author="DWSF Team",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
        "pyyaml>=6.0",
        "matplotlib>=3.4.0",
        "sqlalchemy>=1.4.0",
        "pyarrow>=6.0.0",
        "fastparquet>=0.8.0",
        "jinja2>=3.0.0",
        "multiprocessing-logging>=0.3.0",
    ],
    entry_points={
        "console_scripts": [
            "dwsf=main:main",
        ],
    },
    python_requires=">=3.8",
)

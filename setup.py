#!/usr/bin/env python3
"""
AdaptiveScale Networks (ASN) - Setup Script
A comprehensive implementation of adaptive neural network scaling with SVD decomposition,
meta-learning, and reinforcement learning optimization.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adaptive-scale-networks",
    version="1.0.0",
    author="ASN Research Team",
    author_email="research@asn.ai",
    description="AdaptiveScale Networks: Hierarchical SVD-based Neural Network Adaptation with Meta-Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/asn-research/adaptive-scale-networks",
    project_urls={
        "Bug Tracker": "https://github.com/asn-research/adaptive-scale-networks/issues",
        "Documentation": "https://asn-research.github.io/adaptive-scale-networks/",
        "Repository": "https://github.com/asn-research/adaptive-scale-networks",
    },
    packages=find_packages(exclude=["tests*", "experiments*", "scripts*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "pre-commit>=2.20.0",
        ],
        "visualization": [
            "plotly>=5.10.0",
            "kaleido>=0.2.1",
            "seaborn>=0.11.0",
        ],
        "wandb": [
            "wandb>=0.13.0",
        ],
        "full": [
            "plotly>=5.10.0",
            "kaleido>=0.2.1",
            "seaborn>=0.11.0",
            "wandb>=0.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "asn-train=asn.training.pipeline:main",
            "asn-evaluate=asn.evaluation.evaluator:main",
            "asn-benchmark=experiments.scripts.run_benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "asn": ["config/*.yaml", "data/datasets/*.json"],
    },
    zip_safe=False,
)
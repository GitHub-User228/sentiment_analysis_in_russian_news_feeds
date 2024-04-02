"""
Template.py

Purpose:
    Automates the creation of project folders and files for this project.
    This script will generate a predefined project structure with folders and files.
    If a specified file already exists, it won't be overwritten.

Usage:
    Simply run this script in the desired location to scaffold the project structure.
    `python template.py`

Dependencies:
    - os, pathlib, logging
"""

# Required libraries
import os
from pathlib import Path
import logging

# Set up logging to display activities and any potential issues.
# It logs the creation of directories and files.
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Define the root project name. This will be used to name primary project directories.
project_name = "russian_news_sentiment_analysis"

# Define the project structure.
# Each entry in this list represents either a directory or a file.
# Directories will be created first, followed by files.
list_of_files = [
    # git
    ".gitignore",
    "README.md",
    "LICENSE.txt",
    # Source directory with various sub-directories for modular organization
    "src/run_pipeline.py",
    "src/show_results.py",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_preparation.py",
    f"src/{project_name}/components/data_tokenisation.py",
    f"src/{project_name}/components/label_prediction.py",
    f"src/{project_name}/components/evaluation.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/config.yaml",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/dataset.py",
    f"src/{project_name}/params/datasets.yaml",
    f"src/{project_name}/params/models.yaml",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/pipeline.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    # Folder to store files with data to be ingested
    "data",
    # Folder to store validation results locally
    "metrics",
    # Logs
    "src/logs",
    "src/logs/hadoop_files.txt",
    # Docker and requirements for deployment and environment setup
    "spark_notebook/Dockerfile",
    "spark_notebook/requirements.txt",
    "env/hadoop.env",
    "docker-compose.yml",
    # Research directory for exploratory work
    "research"
]

# Iterate through the list and create the folders and files
for file_path_str in list_of_files:
    file_path = Path(file_path_str)

    # If there's a parent directory (i.e., it's not at the root), create it.
    if file_path.parent and not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Creating directory: {file_path.parent}")

    if '.' in file_path_str:

        # Create the file if it doesn't exist or if it exists but is empty.
        if not file_path.exists() or file_path.stat().st_size == 0:
            file_path.touch()
            logging.info(f"Creating empty file: {file_path}")
        else:
            logging.info(f"{file_path.name} already exists")

    else:

        # Creating directory if it does not exits
        if not os.path.exists(file_path_str):
            os.makedirs(file_path_str)
            logging.info(f"Creating directory: {file_path}")
        else:
            logging.info(f"{file_path} directory already exists")
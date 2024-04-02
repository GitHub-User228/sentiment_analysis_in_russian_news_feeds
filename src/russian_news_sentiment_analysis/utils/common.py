from pathlib import Path
from typing import Any, List
import os
import sys
import yaml
import json
import joblib
import socket

from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from typing import Union

from pyspark.sql import SparkSession

import gc
import torch

from russian_news_sentiment_analysis import logger


@ensure_annotations
def init_spark_session(namespace: str='namespace',
                       app_name: str='myapp',
                       driver_cores: str='8',
                       executor_cores: str='8',
                       driver_memory: str='15g',
                       executor_memory: str='15g',
                       memrory_fraction: str='0.7',
                       memory_storageFraction: str='0.3') -> SparkSession:

    local_ip = socket.gethostbyname(socket.gethostname())

    spark = SparkSession\
        .builder\
        .appName(app_name)\
        .config("spark.driver.host", local_ip)\
        .config("spark.driver.memory", driver_memory)\
        .config("spark.executor.memory", executor_memory)\
        .config("spark.memory.fraction", memrory_fraction)\
        .config("spark.memory.storageFraction", memory_storageFraction)\
        .getOrCreate()

    return spark



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a yaml file, and returns a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the yaml file.

    Raises:
        ValueError: If the yaml file is empty.
        e: If any other exception occurs.

    Returns:
        ConfigBox: The yaml content as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        logger.info("Value exception: empty yaml file")
        raise ValueError("yaml file is empty")
    except Exception as e:
        logger.info(f"An exception {e} has occurred")
        raise e



@ensure_annotations
def save_yaml(path: Path, data: dict):
    """
    Save yaml data

    Args:
        path (Path): path to yaml file
        data (dict): data to be saved in yaml file
    """
    try:
        with open(path, "w") as f:
            yaml.dump(data, f, indent=4)
        logger.info(f"yaml file saved at: {path}")
    except PermissionError:
        logger.error(f"Permission denied to write to {path}")
        raise
    except OSError as e:
        logger.error(f"Failed to save yaml to {path}. Error: {e}")
        raise
        


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"json file saved at: {path}")
    except PermissionError:
        logger.error(f"Permission denied to write to {path}")
        raise
    except OSError as e:
        logger.error(f"Failed to save json to {path}. Error: {e}")
        raise



@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    try:
        with open(path, "r") as f:
            content = json.load(f)
        logger.info(f"json file loaded successfully from: {path}")
        return ConfigBox(content)
    except FileNotFoundError:
        logger.error(f"File not found at {path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied to read {path}")
        raise
    except OSError as e:
        logger.error(f"Failed to load json from {path}. Error: {e}")
        raise



#@ensure_annotations
def load_txt(path: Path) -> list:
    """
    Loads txt file

    Args:
        path (Path): path to txt file

    Returns:
        list: data as a list
    """
    try:
        with open(path, "r") as f:
            content = f.read().split('\n')
        logger.info(f"txt file loaded successfully from: {path}")
        return content
    except FileNotFoundError:
        logger.error(f"File not found at {path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied to read {path}")
        raise
    except OSError as e:
        logger.error(f"Failed to load txt from {path}. Error: {e}")
        raise



@ensure_annotations
def save_txt(path: Path, data: list):
    """
    Save txt data

    Args:
        path (Path): path to txt file
        data (list): data to be saved in txt file
    """
    try:
        with open(path, "w") as f:
            f.write('\n'.join(sorted(data)))
        logger.info(f"txt file saved at: {path}")
    except PermissionError:
        logger.error(f"Permission denied to write to {path}")
        raise
    except OSError as e:
        logger.error(f"Failed to save txt to {path}. Error: {e}")
        raise



def check_if_line_exists(path: Path, line: str):
    """
    Checks if line exists in the txt file

    Args:
        path (Path): path to txt file
        line (str): line to be checked
    """   
    
    data = load_txt(path)
    return line in data


    
def update_txt(path: Path, data: list):
    """
    Save txt data

    Args:
        path (Path): path to txt file
        data (list): new data to be saved in txt file
    """
    
    old_data = load_txt(path)
    save_txt(path, sorted(old_data + data))


def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()
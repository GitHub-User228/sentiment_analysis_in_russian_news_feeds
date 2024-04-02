import os
from pathlib import Path

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from russian_news_sentiment_analysis import logger
from russian_news_sentiment_analysis.utils.common import check_if_line_exists, update_txt
from russian_news_sentiment_analysis.entity.config_entity import DataIngestionConfig



class DataIngestion:
    """
    Class for data ingestion stage.

    Attributes:
    - config (DataIngestionConfig): Configuration settings.
    - path (dict): Dictionary with path values.
    - general_config (dict): Dictionary with the general configuration parameters.
    """
    
    
    def __init__(self, 
                 config: DataIngestionConfig,
                 path: dict,
                 general_config: dict):
        """
        Initializes the DataIngestion component with the given configuration.

        Parameters:
        - config (DataIngestionConfig): Configuration parameters.
        - path (dict): Dictionary with path values
        - general_config (dict): Dictionary with the general configuration parameters.
        """
        
        self.config = config
        self.path = path
        self.general_config = general_config


    def read_data_from_local(self, 
                             filename: str,
                             text_col: str = None,
                             label_col: str = None,
                             reading_kwargs: dict = {}) -> pd.DataFrame:
        """
        Reads data from local and returns it as a pandas DataFrame.
        Also changes names of text_col and label_col columns to the default name

        Parameters:
        - filename (str): Name of the csv or json file with the data to be ingested.
        - text_col (str): Name of the column (attribute) with text.
        - label_col (str): Name of the column (attribute) with labels.
        - reading_kwargs (dict): Dictionary with arguments used when reading data.

        Returns:
        - df (pd.DataFrame): Pandas DataFrame containing the read data.
        """
        
        if text_col == None: text_col = self.config.text_col
        if label_col == None: label_col = self.config.label_col

        format = filename.split('.')[-1]
        if format == 'csv':
            df = pd.read_csv(os.path.join(self.path.data_to_ingest, filename), usecols=[text_col, label_col], **reading_kwargs)
        elif format in ['json', 'jsonl']:
            df = pd.read_json(os.path.join(self.path.data_to_ingest, filename), **reading_kwargs)
            df = df[[text_col, label_col]]
        else:
            logger.error(f"File with the data must be csv or json(l), not {format}")
            raise 

        df = df.rename(columns={text_col: self.general_config.default_text_col, 
                                label_col: self.general_config.default_label_col})

        logger.info(f"Part1. Data has been read")
        
        return df


    def convert_labels(self,
                       df: pd.DataFrame,
                       label_converter: dict = None) -> pd.DataFrame:
        """
        Converts initial labels according to label_converter dictionary.
        Checks if resulting labels are only 0 and 1.

        Parameters:
        - df (pd.DataFrame): Pandas DataFrame.
        - label_converter (dict): Dictionary to convert initial labels into 0 and 1.

        Returns:
        - df (pd.DataFrame): Pandas DataFrame with checked and (optionally) converted labels.
        """
        
        if label_converter != None:
            if not all([v in [0, 1] for v in label_converter.values()]):
                logger.error(f"Dictionary to convert labels has values that are not 0 and 1")
                raise
            df[self.general_config.default_label_col] = df[self.general_config.default_label_col].map(label_converter)
            logger.info(f"Part2. Labels have been converted")
        else:
            labels = df[self.general_config.default_label_col].unique()
            if not all([v in [0, 1] for v in labels]):
                logger.error(f"Data contains labels that are not 0 and 1")
                raise
            logger.info(f"Part2. Labels have not been changed as it was not required")
            
        return df


    def remove_duplicates(self,
                          df: pd.DataFrame,
                          drop_duplicates: bool = False) -> pd.DataFrame:
        """
        Removes duplicates in the data if needed.
        Also creates id column before removing duplicates.

        Parameters:
        - df (pd.DataFrame): Pandas DataFrame.
        - drop_duplicates (bool): Whether to drop duplicates in the data.

        Returns:
        - df (pd.DataFrame): Pandas DataFrame with optionally removed duplicates.
        """

        df['id'] = df.index
                          
        if drop_duplicates: 
            df = df.drop_duplicates(subset=[self.general_config.default_text_col], keep='first')
            logger.info(f"Part3. Duplicates have been dropped")
        else:
            logger.info(f"Part3. Duplicates have not been dropped as it was not required")
        
        return df
        
        
    def save_data_to_hdfs(self, 
                          df: DataFrame,
                          filename: str,
                          spark: SparkSession):
        """
        Saves Spark DataFrame to HDFS.

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - filename (str): Name of the csv or json file with the data to be ingested.
        - spark (SparkSession): Active SparkSession for data processing.
        """

        df = spark.createDataFrame(df)
        logger.info(f"Part4. Spark DataFrame has been created from Pandas DataFrame")   

        df.write.mode('overwrite').format('parquet').save(os.path.join(self.path.raw_data, f"{'.'.join(filename.split('.')[:-1])}.parquet"))
        logger.info(f"Part5. Data has been saved to HDFS")
            
            
    def run_stage(self, 
                  spark: SparkSession,
                  filename: str,
                  text_col: str = None,
                  label_col: str = None,
                  label_converter: dict = None,
                  drop_duplicates: bool = False,
                  reading_kwargs: dict = {}):
        """
        Runs data ingestion stage.

        Parameters:
        - spark (SparkSession): Active SparkSession for data processing.
        - filename (str): Name of the csv or json file with the data to be ingested.
        - text_col (str): Name of the column (attribute) with text.
        - label_col (str): Name of the column (attribute) with labels.
        - label_converter (dict): Dictionary to convert initial labels into 0 and 1.
        - drop_duplicates (bool): Whether to drop duplicates in the data.
        - reading_kwargs (dict): Dictionary with arguments used when reading data.
        """
        
        if check_if_line_exists(Path(self.path.hadoop_files_checklist), os.path.join(self.path.raw_data, f"{'.'.join(filename.split('.')[:-1])}.parquet")):
            
            logger.info(f"=== SKIPPING DATA INGESTION STAGE for the {filename} AS INGESTED DATA ALREADY EXISTS ===")
            
        else:
        
            logger.info(f"=== STARTING DATA INGESTION STAGE for the {filename} data ===")
            
            df = self.read_data_from_local(filename=filename, text_col=text_col, label_col=label_col, reading_kwargs=reading_kwargs)
    
            df = self.convert_labels(df=df, label_converter=label_converter)
    
            df = self.remove_duplicates(df=df, drop_duplicates=drop_duplicates)

            df = df.dropna(subset=[self.general_config.default_text_col, self.general_config.default_label_col])
            
            self.save_data_to_hdfs(df=df, spark=spark, filename=filename)
            
            update_txt(Path(self.path.hadoop_files_checklist), [os.path.join(self.path.raw_data, f"{'.'.join(filename.split('.')[:-1])}.parquet")])
    
            logger.info(f"=== FINISHED DATA INGESTION STAGE for the {filename} data ===")
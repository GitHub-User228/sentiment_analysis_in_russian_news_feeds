import os
import re
import sys
import nltk
import unicodedata
from pathlib import Path
from tqdm.notebook import tqdm
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.column import Column
from pyspark.sql.types import StringType

from russian_news_sentiment_analysis import logger
from russian_news_sentiment_analysis.utils.common import check_if_line_exists, update_txt
from russian_news_sentiment_analysis.entity.config_entity import DataPreparationConfig



nltk.download('stopwords')
nltk.download("wordnet")
nltk.download('omw-1.4')
CLEANER = re.compile('<.*?>')
STOP_WORDS = stopwords.words("russian")
LEMMATIZER = MorphAnalyzer()



class DataPreparation:
    """
    Class for data preparation.

    Attributes:
    - config (DataPreparationConfig): Configuration settings.
    - path (dict): Dictionary with path values
    - general_config (dict): Dictionary with the general configuration parameters.
    """
    
    def __init__(self, 
                 config: DataPreparationConfig, 
                 path: dict,
                 general_config: dict):
        """
        Initializes the DataPreparation component.
        
        Parameters:
        - config (DataPreparationConfig): Configuration settings.
        - path (dict): Dictionary with path values
        - general_config (dict): Dictionary with the general configuration parameters.
        """
        
        self.config = config
        self.path = path
        self.general_config = general_config


    def read_data_from_hdfs(self, 
                            filename: str,
                            spark: SparkSession) -> DataFrame:
        """
        Reads the data from HDFS using the provided SparkSession.

        Parameters:
        - spark (SparkSession): SparkSession object.
        - filename (str): Name of the file to be read (without format specified).

        Returns:
        - data (DataFrame): Spark DataFrame containing the data.
        """
        
        try:
            data = spark.read.parquet(os.path.join(self.path.raw_data, f'{filename}.parquet'))
            logger.info(f"Part1. Raw data has been read from HDFS")
            return data
        except Exception as e:
            logger.error(f"Failed to read data from HDFS. Error: {e}")
            raise e


    def save_data_to_hdfs(self, 
                          data: DataFrame,
                          filename: str,
                          spark: SparkSession):
        """
        Saves the data to HDFS using the provided SparkSession.

        Parameters:
        - data (DataFrame): Spark DataFrame containing the data.
        - filename (str): Name of the file to be used to save the data (without format specified).
        - spark (SparkSession): SparkSession object.
        """
        
        try:
            data.write.mode('overwrite').format('parquet').save(os.path.join(self.path.prepared_data, f'{filename}.parquet'))
            logger.info(f"Part3. Prepared data has been saved to HDFS")
        except Exception as e:
            logger.error(f"Failed to save data to HDFS. Error: {e}")
            raise e


    def prepare_data(self, 
                     data: DataFrame,
                     spark: SparkSession) -> DataFrame:
        """
        Performs preparation of the raw data.

        Parameters:
        - data (DataFrame): Spark DataFrame containing raw data.
        - spark (SparkSession): SparkSession object.

        Returns: 
        - data (DataFrame): Spark DataFrame containing prepared data
        """ 

        @F.udf(returnType=StringType())
        def preprocess_text(input_text: str) -> str:
            """
            Performs preprocessing of input_text string.
            Preprocessing parts:
                - removing control characters
                - removing tags
                - removing extra spacing
                - converting characters to lower case
                - replacing some specific characters with appropriate ones
                - keeping only alphabetic and numeric symbols
                - removing extra spacing
                - removing stop words
                - lemmatazing
        
            Parameters:
            - input_text (str): Text to be prepared
        
            Returns:
            - output_text (str): Prepared text
            """
            
            ### removing control characters
            output_text = "".join(ch if unicodedata.category(ch)[0]!="C" else ' ' for ch in input_text)
            output_text = unicodedata.normalize("NFKD", output_text)
            
            ### removing tags
            output_text = re.sub(CLEANER, ' ', output_text)
            
            ### removing extra spacing
            output_text = re.sub(' +', ' ', output_text).lstrip()
    
            ### to lower case
            output_text = input_text.lower()
            
            ### dealing with specific characters
            output_text = output_text.replace('%', ' процент ')
            output_text = output_text.replace('&quot', ' ')
            output_text = re.sub('й', 'й', output_text)
            
            ### keeping only alphabetic and numeric symbols
            output_text = re.sub("[^ 0-9a-zа-яё]", " ", output_text)
            
            ### removing extra spacing
            output_text = re.sub(' +', ' ', output_text).lstrip()
            
            ### remove stop words
            output_text = [word for word in output_text.split(' ') if (word not in STOP_WORDS) and (word != '')]

            ### cropping text to specified max size
            output_text = output_text[:self.config.max_words]

            ### lemmatazing
            for it in range(len(output_text)):
                try:
                    output_text[it] = LEMMATIZER.normal_forms(output_text[it])[0]
                except:
                    pass

            ### list to string
            output_text = ' '.join(output_text)
            
            return output_text

        # data = data.dropna(subset=[self.general_config.default_text_col])

        # data = data.dropDuplicates(subset=[self.general_config.default_text_col])

        data = data.withColumn(self.general_config.default_text_col, preprocess_text(self.general_config.default_text_col))

        data = data.dropDuplicates(subset=[self.general_config.default_text_col])

        logger.info(f"Part2. Raw data has been prepared")
        
        return data 

    
    def run_stage(self, 
                  spark: SparkSession,
                  filename: str):

        """
        Runs data preparation stage.

        Parameters:
        - spark (SparkSession): SparkSession object.
        - filename (str): Name of the file with data to be read (without format specified).
        """

        # print(check_if_line_exists(Path(self.path.hadoop_files_checklist), os.path.join(self.path.prepared_data, f'{filename}.parquet')))

        if check_if_line_exists(Path(self.path.hadoop_files_checklist), os.path.join(self.path.prepared_data, f'{filename}.parquet')):
            
            logger.info(f"=== SKIPPING DATA PREPARATION STAGE for the {filename} AS PREPARED DATA ALREADY EXISTS ===")
            
        else:

            logger.info(f"=== STARTING DATA PREPARATION STAGE for the {filename} ===")
            
            data = self.read_data_from_hdfs(filename=filename, spark=spark)
    
            data = self.prepare_data(data=data, spark=spark)
    
            self.save_data_to_hdfs(data=data, filename=filename, spark=spark)

            update_txt(Path(self.path.hadoop_files_checklist), [os.path.join(self.path.prepared_data, f'{filename}.parquet')])
    
            logger.info(f"=== FINISHED DATA PREPARATION STAGE for the {filename} ===")
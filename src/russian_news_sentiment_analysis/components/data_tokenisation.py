import os
from pathlib import Path

from transformers import AutoTokenizer

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.column import Column
from pyspark.sql.types import ArrayType, IntegerType, Row, StructField, StructType

from russian_news_sentiment_analysis import logger
from russian_news_sentiment_analysis.utils.common import check_if_line_exists, update_txt
from russian_news_sentiment_analysis.entity.config_entity import DataTokenisationConfig

from hybrid_model_for_russian_sentiment_analysis.model import CustomHybridModel



class DataTokenisation:
    """
    Class for text data tokenisation via specified model's tokeniser.

    Attributes:
    - config (DataTokenisationConfig): Configuration settings.
    - path (dict): Dictionary with path values
    - general_config (dict): Dictionary with the general configuration parameters.
    """
    
    def __init__(self, 
                 config: DataTokenisationConfig, 
                 path: dict,
                 general_config: dict):
        """
        Initializes the DataTokenisation component.
        
        Parameters:
        - config (DataTokenisationConfig): Configuration settings.
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
            data = spark.read.parquet(os.path.join(self.path.prepared_data, f'{filename}.parquet'))
            logger.info(f"Part1. Prepared data has been read from HDFS")
            return data
        except Exception as e:
            logger.error(f"Failed to read data from HDFS. Error: {e}")
            raise e


    def load_tokeniser(self, 
                       model_name: str,
                       is_hugging_face_model: bool,
                       tokeniser_loader_parameters: dict = {}) -> object:
        """
        Function to load a Hugging Face tokeniser

        Paramters:
        - model_name (str): Name of the model which tokeniser will be utilised.
        - is_hugging_face_model (bool): Whether the model_name specifies a model from hugging face
        - tokeniser_loader_parameters (dict): Dictionary with parameters to be used when loading tokeniser.

        Returns:
        - tokeniser (object): Tokeniser object.
        """

        if is_hugging_face_model:
        
            tokeniser = AutoTokenizer.from_pretrained(model_name, **tokeniser_loader_parameters)
            # tokeniser.pad_token = '<pad>'

        elif model_name == 'CustomHybridModel':

            model = CustomHybridModel()
            tokeniser = model.load_tokeniser()

        else:

            logger.error('Can not initialise a tokeniser for the specified model')
            raise

        logger.info(f"Part2. Tokeniser has been loaded")
        
        return tokeniser


    def tokenise_data(self, 
                      data: DataFrame,
                      tokeniser: object) -> DataFrame:
        """
        Function to tokenise text data

        Parameters:
        - data (DataFrame): Spark DataFrame with the text data.
        - tokeniser (object): Tokeniser object.

        Returns:
        - data (DataFrame): Spark DataFrame with omitted text column and new columns: input_ids and attention_masks.
        """

        @F.udf(returnType=StructType([StructField("input_ids", ArrayType(IntegerType())),
                                      StructField("attention_mask", ArrayType(IntegerType()))]))
        def tokenise_text(input_text: str) -> list:
            """
            Performs tokenisation of the input_text string.
        
            Parameters:
            - input_text (str): Text to be tokenised
        
            Returns:
            - output (list): List, where input_ids list is the first value, attention_mask list is the second
            """
            
            output = tokeniser(input_text, **self.config.tokeniser_parameters)
            input_ids = [k for (k, v) in zip(output['input_ids'], output['attention_mask']) if v == 1]
            input_ids = [0 if it >= len(input_ids) else input_ids[it] for it in range(self.config.tokeniser_parameters.max_length)]
            attention_mask = [1 if k!=0 else 0 for k in input_ids]
            output = Row('input_ids', 'attention_mask')(input_ids, attention_mask)
            # output = Row('input_ids', 'attention_mask')(output['input_ids'], output['attention_mask'])
            return output

        data = data.withColumn('output', tokenise_text(self.general_config.default_text_col))
        
        data = data.select('id', 
                           self.general_config.default_label_col, 
                           data['output.input_ids'].alias('input_ids'), 
                           data['output.attention_mask'].alias('attention_mask'))

        logger.info(f"Part3. Data has been tokenised")
        
        return data


    def save_data_to_hdfs(self, 
                          data: DataFrame,
                          filename: str,
                          model_name: str,
                          spark: SparkSession):
        """
        Saves the data to HDFS using the provided SparkSession.

        Parameters:
        - data (DataFrame): Spark DataFrame containing the data.
        - filename (str): Name of the file to be used to save the data (without format specified).
        - model_name (str): Name of the model which tokeniser was utilised.
        - spark (SparkSession): SparkSession object.
        """
        
        try:
            data.write.mode('overwrite').format('parquet').save(os.path.join(self.path.tokenised_data, 
                                                                             f"{filename}_{model_name.replace('/', '--')}.parquet"))
            logger.info(f"Part4. Tokenised data has been saved to HDFS")
        except Exception as e:
            logger.error(f"Failed to save data to HDFS. Error: {e}")
            raise e

    
    def run_stage(self, 
                  spark: SparkSession,
                  filename: str,
                  model_name: str,
                  is_hugging_face_model: bool = True,
                  tokeniser_loader_parameters: dict = {}):
        """
        Runs data tokenisation stage.

        Parameters:
        - spark (SparkSession): SparkSession object.
        - filename (str): Name of the file with data to be read (without format specified).
        - model_name (str): Name of the model which tokeniser will be utilised.
        - is_hugging_face_model (bool): Whether the model_name specifies a model from hugging face
        - tokeniser_loader_parameters (dict): Dictionary with parameters to be used when loading tokeniser.
        """

        if check_if_line_exists(Path(self.path.hadoop_files_checklist), os.path.join(self.path.tokenised_data, 
                                                                             f"{filename}_{model_name.replace('/', '--')}.parquet")):
            
            logger.info(f"=== SKIPPING DATA TOKENISATION STAGE for the {filename} and {model_name} model AS TOKENISED DATA ALREADY EXISTS ===")
            
        else:

            logger.info(f"=== STARTING DATA TOKENISATION STAGE for the {filename} and {model_name} model ===")
            
            data = self.read_data_from_hdfs(filename=filename, spark=spark)
    
            tokeniser = self.load_tokeniser(model_name=model_name, is_hugging_face_model=is_hugging_face_model,
                                            tokeniser_loader_parameters=tokeniser_loader_parameters)
    
            data = self.tokenise_data(data=data, tokeniser=tokeniser)
    
            self.save_data_to_hdfs(data=data, filename=filename, model_name=model_name, spark=spark)
            
            update_txt(Path(self.path.hadoop_files_checklist), 
                       [os.path.join(self.path.tokenised_data, f"{filename}_{model_name.replace('/', '--')}.parquet")])
    
            logger.info(f"=== FINISHED DATA TOKENISATION STAGE for the {filename} and {model_name} model ===")
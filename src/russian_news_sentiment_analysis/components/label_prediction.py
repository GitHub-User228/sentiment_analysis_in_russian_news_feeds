import os
import math
from pathlib import Path
from tqdm.auto import tqdm

import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

from russian_news_sentiment_analysis import logger
from russian_news_sentiment_analysis.entity.dataset import CustomDataset
from russian_news_sentiment_analysis.utils.common import check_if_line_exists, update_txt
from russian_news_sentiment_analysis.entity.config_entity import LabelPredictionConfig

from hybrid_model_for_russian_sentiment_analysis.model import CustomHybridModel



class LabelPrediction:
    """
    Class for making predictions via specified model.

    Attributes:
    - config (LabelPredictionConfig): Configuration settings.
    - path (dict): Dictionary with path values
    - general_config (dict): Dictionary with the general configuration parameters.
    """
    
    def __init__(self, 
                 config: LabelPredictionConfig, 
                 path: dict,
                 general_config: dict):
        """
        Initializes the LabelPrediction component.
        
        Parameters:
        - config (LabelPredictionConfig): Configuration settings.
        - path (dict): Dictionary with path values
        - general_config (dict): Dictionary with the general configuration parameters.
        """
        
        self.config = config
        self.path = path
        self.general_config = general_config


    def read_data_from_hdfs(self, 
                            filename: str,
                            model_name: str,
                            spark: SparkSession) -> DataFrame:
        """
        Reads the data from HDFS using the provided SparkSession.

        Parameters:
        - spark (SparkSession): SparkSession object.
        - model_name (str): Name of the model which is to be utilised.
        - filename (str): Name of the file to be read (without format specified).

        Returns:
        - data (DataFrame): Spark DataFrame containing the data.
        """
        
        try:
            data = spark.read.parquet(os.path.join(self.path.tokenised_data, f"{filename}_{model_name.replace('/', '--')}.parquet"))
            logger.info(f"Part1. Tokenised data has been read from HDFS")
            return data
        except Exception as e:
            logger.error(f"Failed to read data from HDFS. Error: {e}")
            raise e


    def df_to_dataset(self, data: DataFrame) -> CustomDataset:
        """
        Converts Spark DataFrame to CustomDataset

        Parameters:
        - data (DataFrame): Spark DataFrame

        Returns:
        - ds (CustomDataset): Data with input_ids and attention_mask
        - ids (list): List of ids of observations.
        """
        #limit = 322*512
        try:
            ds = Dataset.from_spark(data)
            ids = ds['id']
            #ds = CustomDataset(input_ids=ds['input_ids'][:limit], attention_mask=ds['attention_mask'][:limit])
            ds = CustomDataset(input_ids=ds['input_ids'], attention_mask=ds['attention_mask'])
            logger.info(f"Part2. Data has been transformed to CustomDataset")
            return ds, ids
        except Exception as e:
            logger.error(f"Failed to transform data to CustomDataset. Error: {e}")
            raise e


    def load_model(self, 
                   model_name: str,
                   is_hugging_face_model: bool,
                   device: str) -> object:
        """
        Function to load or initialise a model

        Paramters:
        - model_name (str): Name of the model which will be utilised.
        - is_hugging_face_model (bool): Whether a model is a model from hugging face.
        - device (str): Device on which calculation are to be performed (cuda or cpu)

        Returns:
        - model (object): Model's object.
        """

        if is_hugging_face_model:
            
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            model.eval()
            logger.info(f"Part3. Hugging Face Model has been loaded")

        elif model_name == 'CustomHybridModel':

            model = CustomHybridModel(device=device)
            logger.info(f"Part3. CustomHybridModel has been initialised")

        else:

            logger.error('Can not load or initialise the specified model')
            raise
        
        return model


    def predict(self,
                data: CustomDataset, 
                model: object,
                device: str,
                is_hugging_face_model: bool,
                label_converter: dict = None) -> list[int]:
        """
        Function to encode tokenized data and save resulting embeddings

        Parameters:
        - data (CustomDataset): Tokenized data as CustomDataset.
        - model (object): Model's object.
        - device (str): Device on which calculation are to be performed (cuda or cpu)
        - is_hugging_face_model (bool): Whether a model is a model from hugging face.
        - label_converter (dict): Dictionary to convert labels to the desired two classes (0 and 1)

        Returns:
        - predictions (list): List of predicted labels.
        """

        if is_hugging_face_model:

            # Creating batch generator and tqdm iterator
            batch_generator = torch.utils.data.DataLoader(dataset=data, batch_size=self.config.batch_size, shuffle=False)
            n_batches = math.ceil(len(data)/batch_generator.batch_size)
            iterator = tqdm(enumerate(batch_generator), desc='batch', leave=True, total=n_batches)    
    
            # Making predictions
            with torch.no_grad():
    
                predictions = None
                
                for it, (batch_ids, batch_masks) in iterator:
    
                    # Moving tensors to GPU
                    batch_ids = batch_ids.to(device)
                    batch_masks = batch_masks.to(device)
                
                    # Getting predictions
                    batch_output = torch.argmax(model(input_ids=batch_ids, attention_mask=batch_masks)['logits'], axis=-1)
            
                    predictions = batch_output if predictions==None else torch.cat([predictions, batch_output], axis=0)

        else:

            predictions = model.predict_on_tokens(data=data)
            
        predictions = predictions.tolist()

        if label_converter != None: predictions = [label_converter[v] for v in predictions]
        
        logger.info('Part4. Predictions have been made')
        
        return predictions


    def move_predictions_to_spark(self, 
                                  data: DataFrame,
                                  ids: list,
                                  predictions: list,
                                  spark: SparkSession) -> DataFrame:
        """
        Merges predictions to the existing Spark DataFrame as a new column.

        Parameters:
        - data (DataFrame): Spark DataFrame.
        - ids (list): List of ids of observations.
        - predictions (list): List of predicted labels.

        Returns:
        - data (DataFrame): Spark DataFrame with predictions.
        """
                                  
        predictions = spark.createDataFrame(zip(ids, predictions), ['id', 'prediction'])
        data = data.join(predictions, on='id', how='right')
        data = data.drop('input_ids', 'attention_mask')
        logger.info('Part5. Predictions have been merged')
        
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
        - model_name (str): Name of the model which are to be utilised.
        - spark (SparkSession): SparkSession object.
        """
        
        try:
            data.write.mode('overwrite').format('parquet').save(os.path.join(self.path.predicted_data, 
                                                                             f"{filename}_{model_name.replace('/', '--')}.parquet"))
            logger.info("Part6. Data has been saved to HDFS")
        except Exception as e:
            logger.error(f"Failed to save data to HDFS. Error: {e}")
            raise e


    def run_stage(self, 
                  spark: SparkSession,
                  filename: str,
                  model_name: str,
                  is_hugging_face_model: bool = False,
                  device: str = None,
                  label_converter: dict = None):
        """
        Runs label prediction stage.

        Parameters:
        - spark (SparkSession): SparkSession object.
        - filename (str): Name of the file with data to be read (without format specified).
        - model_name (str): Name of the model which will be utilised.
        - is_hugging_face_model (bool): Whether a model is a model from hugging face.
        - model_config (dict): Dictionary with configuration of the model (only when is_hugging_face_model=False and model_name=CustomHybridModel).
        - device (str): Device on which calculation are to be performed (cuda or cpu)
        - label_converter (dict): Dictionary to convert labels to the desired two classes (0 and 1)
        """
        
        if device == None: device = self.config.device

        if check_if_line_exists(Path(self.path.hadoop_files_checklist), 
                                os.path.join(self.path.predicted_data, f"{filename}_{model_name.replace('/', '--')}.parquet")):
            
            logger.info(f"=== SKIPPING LABEL PREDICTION STAGE for the {filename} and {model_name} model AS PREDICTED DATA ALREADY EXISTS ===")
            
        else:

            logger.info(f"=== STARTING LABEL PREDICTION STAGE for the {filename} and {model_name} model ===")
            
            data = self.read_data_from_hdfs(filename=filename, model_name=model_name, spark=spark)
    
            ds, ids = self.df_to_dataset(data=data)

            model = self.load_model(model_name=model_name, is_hugging_face_model=is_hugging_face_model, device=device)
    
            predictions = self.predict(data=ds, model=model, device=device, is_hugging_face_model=is_hugging_face_model, label_converter=label_converter)
    
            data = self.move_predictions_to_spark(data=data, ids=ids, predictions=predictions, spark=spark)
    
            self.save_data_to_hdfs(data=data, filename=filename, model_name=model_name, spark=spark)

            update_txt(Path(self.path.hadoop_files_checklist), 
                       [os.path.join(self.path.predicted_data, f"{filename}_{model_name.replace('/', '--')}.parquet")])
    
            logger.info(f"=== FINISHED LABEL PREDICTION STAGE for the {filename} and {model_name} model ===")
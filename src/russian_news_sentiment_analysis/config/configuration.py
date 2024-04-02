from russian_news_sentiment_analysis import logger
from russian_news_sentiment_analysis.constants import *
from russian_news_sentiment_analysis.utils.common import read_yaml
from russian_news_sentiment_analysis.entity.config_entity import DataIngestionConfig, DataPreparationConfig
from russian_news_sentiment_analysis.entity.config_entity import DataTokenisationConfig, LabelPredictionConfig, EvaluationConfig



class ConfigurationManager:
    """
    The ConfigurationManager class is responsible for reading and providing 
    configuration settings needed for various stages of the data pipeline.

    Attributes:
    - config (dict): Dictionary holding configuration settings from the config file.
    """
    
    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH):
        """
        Initializes the ConfigurationManager with configurations.

        Parameters:
        - config_filepath (str): Filepath to the configuration file.
        """
        self.config = self._read_config_file(config_filepath, "config")

    
    def _read_config_file(self, filepath: str, config_name: str) -> dict:
        """
        Reads and returns the content of a configuration file.

        Parameters:
        - filepath (str): The file path to the configuration file.
        - config_name (str): Name of the configuration (used for logging purposes).

        Returns:
        - dict: Dictionary containing the configuration settings.

        Raises:
        - Exception: An error occurred reading the configuration file.
        """
        try:
            return read_yaml(filepath)
        except Exception as e:
            logger.error(f"Error reading {config_name} file: {filepath}. Error: {e}")
            raise

            
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Extracts and returns data ingestion configuration settings as a DataIngestionConfig object.

        Returns:
        - DataIngestionConfig: Object containing configuration settings.

        Raises:
        - AttributeError: The 'data_ingestion' attribute does not exist in the config file.
        """
        try:
            config = self.config.data_ingestion
            return DataIngestionConfig(
                text_col=config.text_col,
                label_col=config.label_col
            )
        except AttributeError as e:
            logger.error("The 'data_ingestion' attribute does not exist in the config file.")
            raise e


    def get_data_preparation_config(self) -> DataPreparationConfig:
        """
        Extracts and returns data preparation configuration settings as a DataPreparationConfig object.

        Returns:
        - DataPreparationConfig: Object containing configuration settings.

        Raises:
        - AttributeError: The 'data_preparation' attribute does not exist in the config file.
        """
        try:
            config = self.config.data_preparation
            return DataPreparationConfig(
                max_words=config.max_words
            )
        except AttributeError as e:
            logger.error("The 'data_preparation' attribute does not exist in the config file.")
            raise e


    def get_data_tokenisation_config(self) -> DataTokenisationConfig:
        """
        Extracts and returns data tokenisation configuration settings as a DataTokenisationConfig object.

        Returns:
        - DataTokenisationConfig: Object containing configuration settings.

        Raises:
        - AttributeError: The 'data_tokenisation' attribute does not exist in the config file.
        """
        try:
            config = self.config.data_tokenisation
            return DataTokenisationConfig(
                tokeniser_parameters=config.tokeniser_parameters
            )
        except AttributeError as e:
            logger.error("The 'data_tokenisation' attribute does not exist in the config file.")
            raise e


    def get_label_prediction_config(self) -> LabelPredictionConfig:
        """
        Extracts and returns label prediction configuration settings as a LabelPredictionConfig object.

        Returns:
        - LabelPredictionConfig: Object containing configuration settings.

        Raises:
        - AttributeError: The 'label_prediction' attribute does not exist in the config file.
        """
        try:
            config = self.config.label_prediction
            return LabelPredictionConfig(
                device=config.device,
                batch_size=config.batch_size
            )
        except AttributeError as e:
            logger.error("The 'label_prediction' attribute does not exist in the config file.")
            raise e


    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Extracts and returns evaluation configuration settings as a EvaluationConfig object.

        Returns:
        - EvaluationConfig: Object containing configuration settings.

        Raises:
        - AttributeError: The 'evaluation' attribute does not exist in the config file.
        """
        try:
            config = self.config.evaluation
            return EvaluationConfig(
                metrics_names=config.metrics_names
            )
        except AttributeError as e:
            logger.error("The 'evaluation' attribute does not exist in the config file.")
            raise e
import os
from pathlib import Path
from tqdm.auto import tqdm

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, FloatType, StructType, StructField
from pyspark.ml.evaluation import MultilabelClassificationEvaluator, MulticlassClassificationEvaluator

from russian_news_sentiment_analysis import logger
from russian_news_sentiment_analysis.utils.common import check_if_line_exists, update_txt
from russian_news_sentiment_analysis.entity.config_entity import EvaluationConfig



class Evaluation:
    """
    Class for evaluation of the predictions made by specified model.

    Attributes:
    - config (EvaluationConfig): Configuration settings.
    - path (dict): Dictionary with path values
    - general_config (dict): Dictionary with the general configuration parameters.
    """
    
    def __init__(self, 
                 config: EvaluationConfig, 
                 path: dict,
                 general_config: dict):
        """
        Initializes the Evaluation component.
        
        Parameters:
        - config (EvaluationConfig): Configuration settings.
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
        - model_name (str): Name of the model which was utilised.
        - filename (str): Name of the file to be read (without format specified).

        Returns:
        - data (DataFrame): Spark DataFrame containing the data.
        """
        
        try:
            data = spark.read.parquet(os.path.join(self.path.predicted_data, f"{filename}_{model_name.replace('/', '--')}.parquet"))
            logger.info(f"Part1. Predicted data has been read from HDFS")
            return data
        except Exception as e:
            logger.error(f"Failed to read data from HDFS. Error: {e}")
            raise e


    def evaluate(self, 
                 data: DataFrame,
                 metrics_names: list[str],
                 spark: SparkSession) -> DataFrame:
        """
        Function to encode tokenized data and save resulting embeddings

        Parameters:
        - data (DataFrame): Tokenized data as CustomDataset.
        - metrics_names (list[str]): List of metrics' names to be considered.
        - spark (SparkSession): SparkSession object.

        Returns:
        - metrics (DataFrame): DataFrame with evaluation metrics.
        """

        # 
        data = data.select('id',
                           F.col(self.general_config.default_label_col).cast(DoubleType()),
                           F.col('prediction').cast(DoubleType()))
        
        # Initializationg of the evaluator
        evaluator = MulticlassClassificationEvaluator(labelCol=self.general_config.default_label_col, predictionCol='prediction')
        
        # Calculating metrics
        metrics = {}
        for metric_name in metrics_names: 
            name = f"{metric_name}_1" if 'ByLabel' in metric_name else metric_name
            metrics[name] = evaluator.evaluate(data, {evaluator.metricName: metric_name})

        # Swapping labels
        data = data.withColumn(self.general_config.default_label_col, 1 - F.col(self.general_config.default_label_col)) \
                   .withColumn('prediction', 1 - F.col('prediction'))

        # Calculating metrics
        for metric_name in metrics_names: 
            if 'ByLabel' in metric_name:
                metrics[f"{metric_name}_0"] = evaluator.evaluate(data, {evaluator.metricName: metric_name})

        # Transforming to Spark DataFrame
        schema = StructType([StructField("metric", StringType()), StructField("value", FloatType(), True)])
        metrics = spark.createDataFrame(metrics.items(), schema)

        logger.info("Part2. Metrics have been calculated")

        return metrics
    

    def save_metrics_to_hdfs(self, 
                            data: DataFrame,
                            filename: str,
                            model_name: str,
                            spark: SparkSession):
        """
        Saves the metrics to HDFS using the provided SparkSession.

        Parameters:
        - data (DataFrame): Spark DataFrame containing the data.
        - filename (str): Name of the file to be used to save the data (without format specified).
        - model_name (str): Name of the model which are to be utilised.
        - spark (SparkSession): SparkSession object.
        """
        
        try:
            data.repartition(1).write.mode('overwrite').format('json').save(os.path.join(self.path.metrics_data, 
                                                                             f"{filename}_{model_name.replace('/', '--')}.parquet"))
            logger.info("Part3. Metrics have been saved to HDFS")
        except Exception as e:
            logger.error(f"Failed to save data to HDFS. Error: {e}")
            raise e

    
    def run_stage(self, 
                  spark: SparkSession,
                  filename: str,
                  model_name: str,
                  metrics_names: list[str] = None):
        """
        Runs evaluation stage.

        Parameters:
        - spark (SparkSession): SparkSession object.
        - filename (str): Name of the file with data to be read (without format specified).
        - model_name (str): Name of the model which was utilised.
        - metrics_names (list[str]): List of metrics' names to be considered.
        """

        if metrics_names == None: metrics_names = self.config.metrics_names

        if check_if_line_exists(Path(self.path.hadoop_files_checklist), 
                                os.path.join(self.path.metrics_data, f"{filename}_{model_name.replace('/', '--')}.parquet")):
            
            logger.info(f"=== SKIPPING EVALUATION STAGE for the {filename} and {model_name} model AS METRICS DATA ALREADY EXISTS ===")
            
        else:

            logger.info(f"=== STARTING EVALUATION STAGE for the {filename} and {model_name} model ===")
            
            data = self.read_data_from_hdfs(filename=filename, model_name=model_name, spark=spark)
    
            metrics = self.evaluate(data=data, metrics_names=metrics_names, spark=spark)
    
            self.save_metrics_to_hdfs(data=metrics, filename=filename, model_name=model_name, spark=spark)

            update_txt(Path(self.path.hadoop_files_checklist), 
                       [os.path.join(self.path.metrics_data, f"{filename}_{model_name.replace('/', '--')}.parquet")])
    
            logger.info(f"=== FINISHED EVALUATION STAGE for the {filename} and {model_name} model ===")
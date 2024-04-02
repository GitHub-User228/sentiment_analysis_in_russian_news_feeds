from pathlib import Path

from russian_news_sentiment_analysis import logger
from russian_news_sentiment_analysis.utils.common import init_spark_session, read_yaml
from russian_news_sentiment_analysis.config.configuration import ConfigurationManager

from russian_news_sentiment_analysis.components.data_ingestion import DataIngestion
from russian_news_sentiment_analysis.components.data_preparation import DataPreparation
from russian_news_sentiment_analysis.components.data_tokenisation import DataTokenisation
from russian_news_sentiment_analysis.components.label_prediction import LabelPrediction
from russian_news_sentiment_analysis.components.evaluation import Evaluation



def run(dataset_id: int,
        model_id: int,
        drop_duplicates: bool = True,
        device: str = None,
        last_stage_id: int = 5):
    
    logger.info(f'>>>>>>>>>>>> STARTING PROCESSING dataset with id {dataset_id} >>>>>>>>>>>>')

    # Initialization of the spark session
    spark = init_spark_session()
    logger.info('>>>>>> SPARK SESSION HAS BEEN INITIALIZED >>>>>>')    

    try:
    
        # Initialization of the components
        manager = ConfigurationManager()
        data_ingestion = DataIngestion(config=manager.get_data_ingestion_config(), path=manager.config.path, general_config=manager.config.general)
        data_preparation = DataPreparation(config=manager.get_data_preparation_config(), path=manager.config.path, general_config=manager.config.general)
        data_tokenisation = DataTokenisation(config=manager.get_data_tokenisation_config(), path=manager.config.path, general_config=manager.config.general)
        label_prediction = LabelPrediction(config=manager.get_label_prediction_config(), path=manager.config.path, general_config=manager.config.general)
        evaluation = Evaluation(config=manager.get_evaluation_config(), path=manager.config.path, general_config=manager.config.general)
        logger.info('>>>>>> COMPONENTS HAVE BEEN INITIALIZED >>>>>>')   
    
        # Loading dataset's and model's params
        d_p = read_yaml(Path(manager.config.path.datasets_params))[f'dataset{dataset_id}']
        m_p = read_yaml(Path(manager.config.path.models_params))[f'model{model_id}']
        
        # STAGE 1
        data_ingestion.run_stage(spark=spark, filename=d_p['filename'], text_col=d_p['text_col'], label_col=d_p['label_col'], 
                                 label_converter=d_p['label_converter'], reading_kwargs=d_p['reading_kwargs'], drop_duplicates=drop_duplicates)
        logger.info('>>>>>> I. DATA INGESTION STAGE HAS BEEN COMPLETED >>>>>>')
        
        # STAGE 2
        if last_stage_id >= 2:
            filename = '.'.join(d_p['filename'].split('.')[:-1])
            data_preparation.run_stage(spark=spark, filename=filename)
            logger.info('>>>>>> II. DATA PREPARATION STAGE HAS BEEN COMPLETED >>>>>>')
        else:
            logger.info('>>>>>> II. DATA PREPARATION STAGE HAS BEEN SKIPPED >>>>>>')
        
        # STAGE 3
        if last_stage_id >= 3:
            data_tokenisation.run_stage(spark, filename=filename, model_name=m_p['model_name'], is_hugging_face_model=m_p['is_hugging_face_model'],
                                        tokeniser_loader_parameters=m_p['tokeniser_loader_parameters'])
            logger.info(f'>>>>>> III. DATA TOKENISATION STAGE HAS BEEN COMPLETED for {m_p["model_name"]} model >>>>>>')
        else:   
            logger.info('>>>>>> III. DATA TOKENISATION STAGE HAS BEEN SKIPPED >>>>>>')
        
        # STAGE 4
        if last_stage_id >= 4:
            label_prediction.run_stage(spark, filename=filename, model_name=m_p['model_name'], is_hugging_face_model=m_p['is_hugging_face_model'],
                                       device=device, label_converter=m_p['prediction_converter'])
            logger.info(f'>>>>>> IV. LABEL PREDICTION STAGE HAS BEEN COMPLETED for {m_p["model_name"]} model >>>>>>')   
        else:   
            logger.info('>>>>>> IV. LABEL PREDICTION STAGE HAS BEEN SKIPPED >>>>>>')
    
        # STAGE 5
        if last_stage_id >= 5:
            evaluation.run_stage(spark, filename=filename, model_name=m_p['model_name'])
            logger.info(f'>>>>>> V. EVALUATION STAGE HAS BEEN COMPLETED for {m_p["model_name"]} model >>>>>>') 
        else:   
            logger.info('>>>>>> V. EVALUATION STAGE STAGE HAS BEEN SKIPPED >>>>>>')

        spark.stop()
        logger.info('>>>>>>>>>>>> END OF PROCESSING >>>>>>>>>>>>')

    except Exception as e:
        spark.stop()
        logger.error(f"Pipeline has been stopped due to an exception")
        raise e
from dataclasses import dataclass



@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for the data ingestion process.
    
    Attributes:
    - text_col (str): Default initial name of the column (attribute) with text.
    - label_col (str): Default initial name of the column (attribute) with labels.
    """
    
    text_col: str
    label_col: str



@dataclass(frozen=True)
class DataPreparationConfig:
    """
    Configuration for the data preparation process.
    
    Attributes:
    - max_words (int): Maximum number of words to keep in text.
    """
    
    max_words: int



@dataclass(frozen=True)
class DataTokenisationConfig:
    """
    Configuration for the data tokenisation process.
    
    Attributes:
    - tokeniser_parameters (dict): Dictionary with parameters to be used when tokenising data
    """
    
    tokeniser_parameters: dict



@dataclass(frozen=True)
class LabelPredictionConfig:
    """
    Configuration for the label prediction process.
    
    Attributes:
    - device (str): Device on which calculation are to be performed (cuda or cpu)
    - batch_size (int): Batch size.
    """
    
    device: str
    batch_size: int



@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration for the evaluation process.
    
    Attributes:
    -  metrics_names (list[str]): List of metrics' names to be considered.
    """
    
    metrics_names: list[str]
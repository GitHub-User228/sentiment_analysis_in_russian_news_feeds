import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from russian_news_sentiment_analysis.utils.common import init_spark_session, load_txt
from russian_news_sentiment_analysis.config.configuration import ConfigurationManager



manager = ConfigurationManager()
spark = init_spark_session()

files = [k.split('/')[-1] for k in load_txt(Path(manager.config.path.hadoop_files_checklist)) if 'metrics/' in k]
metrics = [spark.read.json(os.path.join(manager.config.path.metrics_data, file)) for file in files]
files = ['.'.join(k.split('.')[:-1]) for k in files]
datasets = [k.split("_")[0] for k in files]
models = ['_'.join(k.split("_")[1:]) for k in files]

print('-'*100)
old_dataset = datasets[0]

for (dataset, model, metric) in zip(datasets, models, metrics):
    if dataset != old_dataset:
        print('-'*100)
    print(f'DATASET: {dataset}  /  MODEL: {model}')
    metric.show()
    old_dataset = dataset

spark.stop()
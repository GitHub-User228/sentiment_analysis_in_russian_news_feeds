import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

from russian_news_sentiment_analysis import logger
from russian_news_sentiment_analysis.pipelines.pipeline import run



parser = argparse.ArgumentParser()
parser.add_argument('--datasets-ids', type=lambda arg: list(map(int, arg.split(','))))
parser.add_argument('--models-ids', type=lambda arg: list(map(int, arg.split(','))))
parser.add_argument('--device', type=str)
parser.add_argument('--drop-duplicates', type=bool)
parser.add_argument('--last-stage-id', type=int)



args = parser.parse_args()

# if (len(args.datasets_ids) > 1) and (len(args.models_ids) > 1):
#     logger.error(f"An array of ids is specified for both datasets and models")
#     raise
# else:
for dataset_id in args.datasets_ids:
    for model_id in args.models_ids:
        run(dataset_id=dataset_id, model_id=model_id, device=args.device, drop_duplicates=args.drop_duplicates, last_stage_id=args.last_stage_id)
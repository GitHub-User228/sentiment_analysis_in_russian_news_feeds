FROM jupyter/pyspark-notebook

COPY requirements.txt ./

RUN pip3 install -r requirements.txt
RUN gdown --id 12b4AvKwH5NHrzJoz6-nKvJISYr8buPpk
RUN pip3 install hybrid_model_for_russian_sentiment_analysis-0.0.2-py3-none-any.whl

ENTRYPOINT tini -g -- jupyter notebook --allow-root
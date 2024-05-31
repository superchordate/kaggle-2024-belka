# https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook

import os
import polars as pl

os.system("gsutil cp gs://kaggle-417721/train.parquet train.parquet")

dt = pl.read_parquet("train.parquet")

dt.sample(100000).write_parquet("trainsample.parquet")

os.system("gsutil cp sample.parquet gs://kaggle-417721/sample.parquet")

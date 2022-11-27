# This is a sample Python script.
import random

from pyspark.sql import SparkSession
from src.dataLoader import dataLoader
from src.splitting import splitting
from src.preprocessing.Sampling import Sampling
from src.preprocessing.Kcore import Kcore
from src.model import trainModel,evaluateModel, top_movies
from pyspark.sql.functions import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_location = "data/movie_ratings_df.csv"
    file_type = "csv"
    infer_schema = "false"
    first_row_is_header = "true"
    delimiter = ","
    seed = 42
    spark = SparkSession.builder.appName('RecommendationPipeline_Cornacchia_Malitesta').getOrCreate()

    df, itemMap = dataLoader(spark, file_location, file_type, infer_schema, first_row_is_header, delimiter)

    df = Sampling('user',df,0.8,seed)

    df = Kcore(df, K=10)

    train, test = splitting(df,type='User-Hold-Out',seed=seed,split=0.8, Session = spark)

    rec = trainModel(train, model='ALS')

    tabular = evaluateModel(df, train, test, rec, topK=100, session=spark, backend='pandas')

    user = random.choice(df.select('userId').distinct().collect())[0]

    top_movies(rec_model=rec,model=itemMap,df = df,user_id=user,n=10)

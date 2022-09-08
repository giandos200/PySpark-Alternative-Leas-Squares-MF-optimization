# This is a sample Python script.
from pyspark.sql import SparkSession
from src.dataLoader import dataLoader
from src.splitting import splitting
from src.preprocessing.Sampling import Sampling
from src.preprocessing.Kcore import Kcore
from src.model import trainModel,evaluateModel
from pyspark.sql.functions import *
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_location = "data/movie_ratings_df.csv"
    file_type = "csv"
    infer_schema = "false"
    first_row_is_header = "true"
    delimiter = ","
    seed = 42
    spark = SparkSession.builder.appName('RecommendationPipeline_Cornacchia_Malitesta').getOrCreate()

    df = dataLoader(spark, file_location, file_type, infer_schema, first_row_is_header, delimiter)

    df = Sampling('user',df,0.8,seed)

    df = Kcore(df, K=5)

    train, test = splitting(df,type='LeaveOneOut',seed=42,split=0.8, Session = spark)

    rec = trainModel(train, model='ALS')

    tabular = evaluateModel(df, train, test, rec, topK=10, session=spark)

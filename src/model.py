from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame
from typing import List
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import *

from tqdm import tqdm
from tabulate import tabulate

def trainModel(train, model ='ALS'):
    if model == 'ALS':
        rec = ALS(maxIter=15, regParam=0.01, userCol='userId', itemCol='title_new',
              ratingCol='rating', nonnegative=True, coldStartStrategy="drop")
    else:
        raise NotImplementedError
    return rec.fit(train)

def evaluateModel(df: DataFrame, train: DataFrame, test: DataFrame, model, topK: int, session)-> List:
    results = []
    predicted_ratings = model.transform(test)
    evaluator = RegressionEvaluator(metricName='rmse', predictionCol=
    'prediction', labelCol='rating')
    rmse = evaluator.evaluate(predicted_ratings)
    listTotMovies = df.select('title_new').distinct()
    listUser = df.select('userId').distinct()
    counter = 0
    for u in tqdm(listUser.collect()):
        watched = train.filter(train['userId'] == u[0]).select('title_new').distinct()
        b = watched.alias('b')
        rem_movies = listTotMovies.join(b, listTotMovies.title_new == b.title_new, how='left'). \
            where(col("b.title_new").isNull()).select(listTotMovies.title_new).distinct()
        rem_movies = rem_movies.withColumn("userId", lit(int(u[0])))
        recomm = model.transform(rem_movies).orderBy('prediction',ascending=False)
        recList = [i[0] for i in recomm.limit(topK).select('title_new').collect()]
        testWatched = [i[0] for i in test.filter(test.userId == u[0]).select('title_new').collect()]
        results.append(tuple([recList,testWatched]))
        counter+=1
        if counter==200:
            break
    tabular = []
    prediction = session.sparkContext.parallelize(results)
    metrics = RankingMetrics(prediction)
    for k in range(1,topK+1):
        prec = metrics.precisionAt(k)
        meanAv = metrics.meanAveragePrecisionAt(k)
        recall = metrics.recallAt(k)
        ndcg = metrics.ndcgAt(k)
        tabular.append([rmse,k,ndcg, prec, recall, meanAv])
    print(tabulate(tabular, headers=['rmse', 'TopN', 'NDCG', 'Precision', 'Recall', 'MeanAveragePrecision']))



    pass
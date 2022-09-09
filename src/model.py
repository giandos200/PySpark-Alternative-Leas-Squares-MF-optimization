import pandas as pd
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame
from typing import List
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, IndexToString
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

def evaluateModel(df: DataFrame, train: DataFrame, test: DataFrame, model, topK: int, session, backend:str = 'pyspark')-> List:
    results = []
    predicted_ratings = model.transform(test)
    evaluator = RegressionEvaluator(metricName='rmse', predictionCol=
    'prediction', labelCol='rating')
    rmse = evaluator.evaluate(predicted_ratings)
    listTotMovies = df.select('title_new').distinct()
    listUser = df.select('userId').distinct()
    backend = backend.lower()
    if backend in ['pyspark', 'pandas', 'mixed']:
        if backend == 'pyspark':
            for u in tqdm(listUser.collect()):
                watched = train.filter(train['userId'] == u[0]).select('title_new').distinct()
                b = watched.alias('b')
                rem_movies = listTotMovies.join(b, listTotMovies.title_new == b.title_new, how='left'). \
                    where(col("b.title_new").isNull()).select(listTotMovies.title_new).distinct()
                rem_movies = rem_movies.withColumn("userId", lit(int(u[0])))
                recomm = model.transform(rem_movies).orderBy('prediction', ascending=False)
                recList = [i[0] for i in recomm.limit(topK).select('title_new').collect()]
                testWatched = [i[0] for i in test.filter(test.userId == u[0]).orderBy('rating', ascending=False).select(
                    'title_new').collect()]
                results.append(tuple([recList, testWatched]))
        else:
            totMovie = set([i[0] for i in listTotMovies.collect()])
            dataframePrediction = {'userId': [], 'title_new': []}
            for u, dframe in train.toPandas().groupby('userId'):
                l = list(totMovie.difference(dframe['title_new'].unique().tolist()))
                dataframePrediction['title_new'].extend(l)
                dataframePrediction['userId'].extend([u for _ in l])
            dataframePrediction = pd.DataFrame(dataframePrediction)
            data = session.createDataFrame(dataframePrediction)
            predictions = model.transform(data)
            if backend == 'pandas':
                p = predictions.toPandas()
                TEST = test.toPandas()
                for u in tqdm(listUser.collect()):
                    recList = p[p['userId'] == u[0]].sort_values(by='prediction', ascending=False)[
                        'title_new'].to_list()
                    testWatched = TEST[TEST['userId'] == u[0]].sort_values(by='rating', ascending=False)[
                        'title_new'].to_list()
                    results.append(tuple([recList, testWatched]))
            elif backend == 'mixed':
                for u in tqdm(listUser.collect()):
                    recList = [i[0] for i in
                               predictions.filter(predictions['userId'] == u[0]).orderBy('prediction').select(
                                   'title_new').collect()]
                    testWatched = [i[0] for i in
                                   test.filter(test['userId'] == u[0]).orderBy('rating').select('title_new').collect()]
                    results.append(tuple([recList, testWatched]))
        tabular = []
        prediction = session.sparkContext.parallelize(results)
        metrics = RankingMetrics(prediction)
        for k in range(1, topK + 1):
            prec = metrics.precisionAt(k)
            meanAv = metrics.meanAveragePrecisionAt(k)
            recall = metrics.recallAt(k)
            ndcg = metrics.ndcgAt(k)
            tabular.append([rmse, k, ndcg, prec, recall, meanAv])
        print(tabulate(tabular, headers=['rmse', 'TopN', 'NDCG', 'Precision', 'Recall', 'MeanAveragePrecision']))
    else:
        raise ValueError('The Backend chosen does not exist! Chose one between pyspark, pandas, and mixed!')
    # #PANDAS BACKEND ENGINE
    # totMovie = set([i[0] for i in listTotMovies.collect()])
    # dataframePrediction = {'userId':[], 'title_new':[]}
    # for u, dframe in train.toPandas().groupby('userId'):
    #     l = list(totMovie.difference(dframe['title_new'].unique().tolist()))
    #     dataframePrediction['title_new'].extend(l)
    #     dataframePrediction['userId'].extend([u for _ in l])
    # dataframePrediction = pd.DataFrame(dataframePrediction)
    # data = session.createDataFrame(dataframePrediction)
    # predictions = model.transform(data)
    # p = predictions.toPandas()
    # TEST = test.toPandas()
    # for u in tqdm(listUser.collect()):
    #     recList = p[p['userId'] == u[0]].sort_values(by='prediction', ascending=False)['title_new'].to_list()
    #     testWatched = TEST[TEST['userId']==u[0]].sort_values(by='rating',ascending=False)['title_new'].to_list()
    #     #testWatched = [i[0] for i in
    #     #               test.filter(test['userId'] == u[0]).orderBy('rating').select('title_new').collect()]
    #     results.append(tuple([recList, testWatched]))
    #
    # for u in tqdm(listUser.collect()):
    #     recList = [i[0] for i in predictions.filter(predictions['userId']==u[0]).orderBy('prediction').select('title_new').collect()]
    #     testWatched = [i[0] for i in test.filter(test['userId'] == u[0]).orderBy('rating').select('title_new').collect()]
    #     results.append(tuple([recList, testWatched]))
    # tabular = []
    # prediction = session.sparkContext.parallelize(results)
    # metrics = RankingMetrics(prediction)
    # for k in range(1,topK+1):
    #     prec = metrics.precisionAt(k)
    #     meanAv = metrics.meanAveragePrecisionAt(k)
    #     recall = metrics.recallAt(k)
    #     ndcg = metrics.ndcgAt(k)
    #     tabular.append([rmse,k,ndcg, prec, recall, meanAv])
    # print(tabulate(tabular, headers=['rmse', 'TopN', 'NDCG', 'Precision', 'Recall', 'MeanAveragePrecision']))
    #

def top_movies(rec_model: object, model: object, df: "DataFrame", user_id: int, n: int):
    """
    This function returns the top 'n' movies that user has not seen yet but
    might like
    """
    # assigning alias name 'a' to unique movies df
    unique_movies=df.select('title_new').distinct()
    a = unique_movies.alias('a')
    # creating another dataframe which contains already watched movie by active user
    watched_movies = df.filter(df['userId'] == user_id).select('title_new')
    # assigning alias name 'b' to watched movies df
    b = watched_movies.alias('b')
    # joining both tables on left join
    total_movies = a.join(b, a.title_new == b.title_new, how='left')
    # selecting movies which active user is yet to rate or watch
    remaining_movies = total_movies.where(col("b.title_new").isNull()).select(a.title_new).distinct()
    # adding new column of user_Id of active useer to remaining movies df
    remaining_movies = remaining_movies.withColumn("userId", lit(int(user_id)))
    # making recommendations using ALS recommender model and selecting only top 'n' movies
    recommendations = rec_model.transform(remaining_movies).orderBy('prediction', ascending=False).limit(n)
    # adding columns of movie titles in recommendations
    movie_title = IndexToString(inputCol="title_new",
                                outputCol="title", labels=model.labels)
    final_recommendations = movie_title.transform(recommendations)
    output = final_recommendations.select('userId', 'title_new', 'title')
    # return the recommendations to active user
    return output.show(n, False)
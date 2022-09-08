import random
from typing import Tuple

from pyspark.sql.dataframe import DataFrame


def splitting(df: DataFrame, type: str, seed: int, split=0.8, Session = None)-> Tuple["DataFrame", "DataFrame"]:
    if type in ["Hold-Out", "LeaveOneOut", "Temporal-LeaveOneOut"]:
        if type == 'Hold-Out':
            train, test = df.randomSplit([split, 1-split],seed=seed)
            test = test.orderBy('rating',ascending=False)
            return train, test
        elif type == 'LeaveOneOut':
            data = df.toPandas()
            test = data.groupby('userId').sample(n=1)
            test['user'] = test['userId']
            test['item'] = test['title_new']
            test.drop(columns=['userId', 'title', 'rating', 'title_new'], inplace=True)
            # l = df.select('userId').distinct().collect()
            # print("\nStarting Leave-One-Out splitting methodology...\n")
            # test = [[],[]]
            # for i in tqdm(l):
            #     itemBag = df.filter(df.userId==i[0]).select('title_new').collect()
            #     test[0].append(i[0])
            #     test[1].append(random.choice(itemBag)[0])
            testDf = Session.createDataFrame(test)
            train = df.join(testDf,(df.userId==testDf.user) & (df.title_new!=testDf.item), 'inner').drop('user','item')
            test = df.join(testDf,(df.userId==testDf.user) & (df.title_new==testDf.item), 'inner').drop('user','item')
            return train, test #nel join si perde inspiegabilmente 9 interazioni
        elif type == 'Temporal-LeaveOneOut':
            raise NotImplementedError
    else:
        raise ValueError(f"The splitting type {type} does not exit! Chose one between Hold-Out, LeaveOneOut, and Temporal-LeaveOneOut")
    pass
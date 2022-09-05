from pyspark.sql.dataframe import DataFrame

def KcoreUser(data: DataFrame, K: int)-> "DataFrame":
    pass

def KcoreItem(data: DataFrame, K: int)-> "DataFrame":
    pass

def Kcore(data: DataFrame, K: int)-> "DataFrame":
    end = False
    while not end:
        datau = KcoreUser(data, K)
        datai = KcoreItem(data, K)
        if datau.count()  == datai.count():
            end = True
        else:
            datau = datai
    return datai
from pyspark.sql.dataframe import DataFrame

def KcoreUser(data: DataFrame, K: int)-> "DataFrame":
    userK = data.groupBy('userId').count()
    userK = userK.filter(userK[1] > 100).select('userId').withColumnRenamed('userId','u_userId')
    data = data.join(userK, data.userId == userK.u_userId , how='inner').drop('u_userId')
    return data

def KcoreItem(data: DataFrame, K: int)-> "DataFrame":
    itemK = data.groupBy('title_new').count()
    itemK = itemK.filter(itemK[1] > 100).select('title_new').withColumnRenamed('title_new','i_title_new')
    data = data.join(itemK, data.title_new == itemK.i_title_new , how='inner').drop('i_title_new')
    return data


def Kcore(data: DataFrame, K: int)-> "DataFrame":
    counter = 0
    while True:
        counter += 1
        datau = KcoreUser(data, K)
        datai = KcoreItem(datau, K)
        data = datai
        if datau.count()  == datai.count():
            print(f"Number of K_core_user and K_core_item filtering iteration: {counter}")
            return data
        else:
            if counter>100:
                raise ValueError("K core value is too large, try to reduce it!")
            continue
from pyspark.sql.dataframe import DataFrame

def Sampling(type: str, data: DataFrame, percentage: int, seed: int)-> "DataFrame":
    if type in ["user", "item", "interactions"]:
        if type == "interactions":
            data = data.sample(withReplacement=False,fraction=percentage, seed=seed)
        elif type == "user":
            user = data.select('userId').distinct().sample(
                withReplacement=False,
                fraction=percentage,
                seed=seed).withColumnRenamed('userId','u_userId')
            data = data.join(user, data.userId == user.u_userId , how='inner').drop('u_userId')
        elif type == "item":
            item =  data.select('title_new').distinct().sample(
                withReplacement=False,
                fraction=percentage,
                seed=seed).withColumnRenamed('title_new','i_title_new')
            data = data.join(item, data.title_new == item.i_title_new, how='inner').drop('i_title_new')
        return data
    else:
        raise NotImplementedError

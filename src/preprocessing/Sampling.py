from pyspark.sql.dataframe import DataFrame

def Sampling(type: str, data: DataFrame, percentage: int, seed: int)-> "DataFrame":
    if type in ["user", "item", "interactions"]:
        if type == "interactions":
            data = data.sample(withReplacement=False,fraction=percentage, seed=seed)
        elif type == "user":
            user = data.select('userId').distinct().sample(
                withReplacement=False,
                fraction=percentage,
                seed=seed)
            data = data.join(user, data.userId == user.userId , how='inner')
        elif type == "item":
            item =  data.select('title_new').distinct().sample(
                withReplacement=False,
                fraction=percentage,
                seed=seed)
            data = data.join(item, data.title_new == item.title_new, how='inner')
        return data
    else:
        raise NotImplementedError

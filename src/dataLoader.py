from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer,IndexToString
from pyspark.sql.types import DoubleType

def dataLoader(Session, path: str, fileType: str, inferSchema: bool, header: bool, sep: str)-> "DataFrame":
    df =  Session.read.format(fileType) \
        .option("inferSchema", inferSchema) \
        .option("header", header) \
        .option("sep", sep) \
        .load(path)

    df.show(5, False)

    print((df.count(), len(df.columns)))

    df.printSchema()

    df.orderBy(rand()).show(10, False)

    df.groupBy('userId').count().orderBy('count', ascending=False).show(10, False)

    df.groupBy('userId').count().orderBy('count', ascending=True).show(10, False)

    df.groupBy('title').count().orderBy('count', ascending=False).show(10, False)

    df = df.withColumn("userId", df.userId.cast(DoubleType()))
    df = df.withColumn("rating", df.rating.cast(DoubleType()))

    stringIndexer = StringIndexer(inputCol="title", outputCol="title_new")

    model = stringIndexer.fit(df)
    indexed = model.transform(df)

    return indexed
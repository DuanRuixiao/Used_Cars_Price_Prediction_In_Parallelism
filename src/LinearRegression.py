import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import math

from csv import reader
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import stddev, mean, col
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.ml.feature import OneHotEncoder, StringIndexer, RegexTokenizer, StopWordsRemover, Word2Vec, VectorAssembler
from pyspark.ml.regression import LinearRegression


# One hot encoder function.
def tfCategoryToBVector(df, inputColumn, outputColumn):
    """
    Given a dataframe df and one inputColumn, convert this column by binarization and output as outputColumn.
    """
    stringIndexer = StringIndexer(inputCol=inputColumn, outputCol=outputColumn + 'Index')
    model = stringIndexer.fit(df)
    indexed = model.transform(df)

    encoder = OneHotEncoder(inputCol=outputColumn + 'Index', outputCol=outputColumn + 'Vec')
    encoded = encoder.transform(indexed)
    return encoded


# Normalization function.
def norm_func(df, input_col):
    """
    Given a dataframe df and onr input_col, apply normalization on all values in this column.

        data_norm = (data - mean) / standard deviation.
    """
    mean_year, stddev_year = df.select(mean(input_col), stddev(input_col)).first()
    df = df.withColumn(input_col + '_norm', (col(input_col) - mean_year) / stddev_year)
    return df


# Split function.
def split_df(df, N):
    num_rows = df.count()
    partition_size = num_rows//N
    train_test = []
    for i in range(N):
        index_start = partition_size*i
        index_end = partition_size*(i+1)
        smaller = df.limit(index_start)
        bigger = df.limit(index_end)
        test_df = bigger.subtract(smaller)
        test_df.cache()
        train_df = df.subtract(test_df)
        train_df.cache()
        train_test.append((train_df, test_df))
    return train_test


# Hyperparameters need to be tuned.
vecSize = 30  # car's brand
lam = 0.3  # [0:1:5] _._
alpha = 1  # [0:1] _._
eps = 1.1  # constant
k_folds = 10

spark = SparkSession.builder \
    .appName("used car price prediction") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# File location and type.
file_location = "train-data.csv"
file_type = "csv"

# The applied options are for CSV files. For other file types, these will be ignored.
train_reviews = spark.read.format(file_type) \
    .option("inferSchema", True) \
    .option("header", True) \
    .load(file_location)

# Take a overview of entire dataset.
# train_reviews.show()

# Data clean with New_Price column and
df = train_reviews.drop('New_Price').na.drop()

print(df.count())

# Preprocess with Name column.
names_only = train_reviews.drop('Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type',
                                'Mileage', 'Engine', 'Power', 'Seats', 'New_Price', 'Price')
namesrdd = names_only.rdd
names_dataframe = spark.createDataFrame(namesrdd)

# Convert Name feature to tokens.
regexTokenizer = RegexTokenizer(gaps=False, pattern='\w+', inputCol='Name', outputCol='name_text_token')
names_token = regexTokenizer.transform(names_dataframe)

# Remove stopwords.
swr = StopWordsRemover(inputCol='name_text_token', outputCol='names_sw_removed')
names_swr = swr.transform(names_token)

# Create an word2Vec converter with dimension vecSize and convert all text to a vecSize-dimension vector.
word2vec = Word2Vec(vectorSize=vecSize, minCount=5, inputCol='names_sw_removed', outputCol='result')
model = word2vec.fit(names_swr)
result = model.transform(names_swr)

# Process with Location, Fuel_Type, Transmission and Owner_Type columns,
# treat theses features as categorical features and apply one-hot encoder.
for features in ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type']:
    temp = tfCategoryToBVector(df, features, features)
    df = temp

# Process with Year, Kilometer_Driven, Mileage, Engine, Power, Seats columns.
# Year, Kilometer_Driven and Seats are numbers, apply normalization.
# Mileage, Engine and Power are numbers with units, ignore the units and keep the value only, then apply normalization.
# Some samples in Power column is "null bpm", so apply a filter to remove these samples.
result_train = df.drop('Name', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'LocationIndex',
                       'Fuel_TypeIndex', 'TransmissionIndex', 'Owner_TypeIndex')
result_train = result_train.join(result.select(['_c0', 'result']), '_c0').withColumnRenamed('result', 'Name_Vec')
label = result_train.select('Price')

result_train_rdd = result_train.rdd
result_train_rdd = result_train_rdd.map(list) \
    .map(lambda li: li[:3] + [eval(li[3].split(' ', 1)[0])] + [eval(li[4].split(' ', 1)[0])] + [li[5].split(' ', 1)[0]]
                    + [li[6]] + li[8].toArray().tolist() + li[9].toArray().tolist() + li[10].toArray().tolist()
                    + li[11].toArray().tolist() + li[12:] + [li[7]]) \
    .filter(lambda li: li[5] != 'null') \
    .map(lambda li: li[:5] + [float(li[5])] + li[6:])

result_train_rdd.collect()
result_train_df = spark.createDataFrame(result_train_rdd) \
                       .withColumnRenamed('_2', 'Year') \
                       .withColumnRenamed('_3', 'Kilometers_Driven') \
                       .withColumnRenamed('_4', 'Mileage') \
                       .withColumnRenamed('_5', 'Engine') \
                       .withColumnRenamed('_6', 'Power') \
                       .withColumnRenamed('_7', 'Seats')

d = result_train_df
for c in ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats']:
    temp = norm_func(d, c)
    d = temp
result_train_df = d

result_train_df = result_train_df.drop('Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats')
# result_train_df.show()

# Concatenate all columns together and use VectorAssembler to generate dataset dataframe.
cat_column_names = ['Year_norm', 'Kilometers_Driven_norm', 'Mileage_norm', 'Engine_norm', 'Power_norm', 'Seats_norm',
                    '_8', '_9', '_10', '_11', '_12', '_13', '_14', '_15', '_16', '_17', '_18', '_19', '_20', '_21',
                    '_22', '_23', '_24', '_25']

cat_assembler = VectorAssembler(inputCols=cat_column_names, outputCol="features")
cat_output = cat_assembler.transform(result_train_df)
cat_output = cat_output.select('features', '_26') \
                       .withColumnRenamed('_26', 'Price')
# cat_output.show(truncate=False)

# sample = cat_output.rdd.take(1)
#
# for s in sample[0]:
#     print(type(s))

# Cross-validation to train model.
# splits = cat_output.randomSplit([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], seed=42)
train_test = split_df(cat_output, k_folds)

train_RMSE = []
test_RMSE = []
for k in range(k_folds):
    # Generate train dataset.
    # df = splits[:k] + splits[k + 1:]
    # train_df = df[0]
    # for d in df[1:]:
    #     train_df.unionAll(d)
    # test_df = splits[k]
    train_df = train_test[k][0]
    test_df = train_test[k][1]

    # Define model by LinearRegression.
    lr = LinearRegression(featuresCol='features', labelCol='Price', maxIter=500, regParam=lam, elasticNetParam=alpha,
                          epsilon=eps)
    lr_model = lr.fit(train_df)

    # Output beta.
    # print("Coefficients: " + str(lr_model.coefficients))
    trainingSummary = lr_model.summary
    print("Train RMSE of %dth fold is: %f" % (k, trainingSummary.rootMeanSquaredError))
    train_RMSE.append(trainingSummary.rootMeanSquaredError)

    # Use model to predict and output test RMSE.
    lr_prediction = lr_model.transform(test_df)
    lr_prediction_rdd = lr_prediction.rdd
    cnt = lr_prediction_rdd.count()
    test_error = lr_prediction_rdd.map(lambda x: (x[2] - x[1]) ** 2) \
                                  .reduce(lambda x, y: x + y)
    test_RMSE.append(math.sqrt(test_error / cnt))

    # Print 10 samples of predictions.
    print('Test RMSE of %dth fold is: %f' % (k, math.sqrt(test_error / cnt)))
    lr_prediction.show(10)

print('The average train RMSEs of cross validation: ', sum(train_RMSE) / len(train_RMSE))
print('The average test RMSEs of cross validation:', sum(test_RMSE) / len(test_RMSE))

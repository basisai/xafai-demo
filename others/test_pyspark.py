#!/usr/bin/env python
# coding: utf-8

# In[1]:


spark


# In[2]:


import numpy as np
import pandas as pd
import shap

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[3]:


# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("libsvm").load("data/mllib_sample_libsvm_data.txt")
data.show(2)


# In[4]:


(trainingData, testData) = data.randomSplit([0.7, 0.3])


# ## LogisticRegression

# In[5]:


lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(trainingData)


# In[6]:


# Make predictions.
predictions = lrModel.transform(testData)

# Select example rows to display.
predictions.show(5)


# In[9]:


print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

np.asarray(lrModel.coefficients)


# In[ ]:





# ## RandomForestClassifier

# In[5]:


rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)

rfModel = rf.fit(trainingData)


# In[14]:


# Make predictions.
predictions = rfModel.transform(testData)

# Select example rows to display.
predictions.show(5)


# In[ ]:





# In[7]:


# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)


# In[8]:


# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)


# In[12]:


predictions.show(5)


# In[9]:


# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only


# ## SHAP

# In[6]:


explainer = shap.TreeExplainer(rfModel)


# In[7]:


vector_udf = F.udf(lambda x: x.toArray().tolist(), ArrayType(DoubleType()))


# In[17]:


aa = testData.select(vector_udf("features").alias("features")).toPandas()
aa["features"] = aa["features"].apply(lambda x: np.asarray(x))
aa = aa.values
aa[0]


# In[29]:


bb = testData.select("features").toPandas()
bb["features"] = bb["features"].apply(lambda x: np.asarray(x))
bb = bb.values
bb[0]


# In[31]:


feats = np.apply_along_axis(lambda x : x[0], 1, bb)
print(feats.shape)
feats


# In[32]:


shap_values = explainer.shap_values(feats, check_additivity=False)


# In[33]:


shap.summary_plot(shap_values)


# ## distributed shap_values

# In[ ]:


X_df = pruned_parsed_df.drop("ConvertedComp").repartition(16)
X_columns = X_df.columns


# In[ ]:


def add_shap(rows):
    rows_pd = pd.DataFrame(rows, columns=X_columns)
    shap_values = explainer.shap_values(rows_pd.drop(["Respondent"], axis=1))
    return [Row(*([int(rows_pd["Respondent"][i])] + [float(f) for f in shap_values[i]])) for i in range(len(shap_values))]


shap_df = X_df.rdd.mapPartitions(add_shap).toDF(X_columns)

effects_df = (
    shap_df
    .withColumn("gender_shap", F.col("Gender_Woman") + F.col("Gender_Man") + F.col("Gender_Non_binary__genderqueer__or_gender_non_conforming") + F.col("Trans"))
    .select("Respondent", "gender_shap")
)

top_effects_df = effects_df.filter(F.abs(F.col("gender_shap")) >= 2500).orderBy("gender_shap")


# In[ ]:


assembler = VectorAssembler(
    inputCols=[c for c in to_review_df.columns if c != "Respondent"],
    outputCol="features",
)
assembled_df = assembler.transform(shap_df).cache()


# In[ ]:


clusterer = BisectingKMeans().setFeaturesCol("features").setK(50).setMaxIter(50).setSeed(0)
cluster_model = clusterer.fit(assembled_df)
transformed_df = cluster_model.transform(assembled_df).select("Respondent", "prediction")


# In[ ]:





# ## Smote

# In[ ]:


from smote import vectorizer_func, smote_sampling


# In[ ]:


dataInput = spark.read.format('csv').options(header='true',inferSchema='true').load("sam.csv").dropna()

df = smote_sampling(vectorizer_func(dataInput, 'Y'),
                    k=2,
                    minority_class=1,
                    majority_class=0,
                    pct_over=90,
                    pct_under=5)


# In[ ]:





# ## Misc

# In[2]:


from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import HiveContext
from pyspark.sql.types import (
    ArrayType, IntegerType, BooleanType, StringType, StructField, StructType)


# In[14]:


# raw_schema = StructType([StructField('age', StringType(), True)])
# df = spark.read.json('people.json', schema=raw_schema)
# df.printSchema()
# df.show()


# In[5]:


spark = SparkSession.builder.appName("PySpark_Testing").getOrCreate()
sc = spark.sparkContext
sqlContext = HiveContext(sc)


# In[6]:


df = spark.createDataFrame([[1,2,3],[2,3,4]], ['fd','dsf','sd'])
df.show()


# In[7]:


sc.getConf().getAll()


# In[ ]:





# In[1]:


spark.stop()


# In[3]:


spark = (
    SparkSession.builder
    .appName("PySpark_Testing")
    .config('spark.driver.memory','4g')
    .config('spark.driver.cores','2')
    .config('spark.executor.instances', '2')
    .config('spark.executor.memory', '4g')
    .config('spark.executor.cores', '2')
    .config("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
    .config('spark.hadoop.fs.AbstractFileSystem.gs.impl', 'com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS')
    .config('spark.hadoop.google.cloud.auth.service.account.enable', 'true')
    .config("fs.gs.auth.service.account.json.keyfile", "span-production-b5d8eee16b8b.json")
    .getOrCreate()
)

spark


# In[4]:


spark.sparkContext.getConf().getAll()


# In[5]:


subscribers = spark.read.parquet("gs://bedrock-sample/churn_data/subscribers.gz.parquet")
subscribers.show(5)


# In[ ]:





# In[41]:


spark.stop()


# In[47]:


sc.getConf().getAll()


# In[ ]:





# In[6]:


def get_stopwords(spark_context, stopwords_file, num=758):
    """Get stopwords.
    There are 758 words in the file."""
    return spark_context.textFile(stopwords_file).take(num)


def get_stopwords_bc(spark_context, stopwords_file, num=758):
    """Get stopwords.
    There are 758 words in the file."""
    stopwords = spark_context.textFile(stopwords_file).take(num)
    return spark_context.broadcast(stopwords)


# In[18]:


with SparkSession.builder.appName("ArticlePreprocessing").getOrCreate() as spark:
#     stopwords = get_stopwords(spark.sparkContext, "./inputs/bahasa_stopwords.txt")
#     print(len(stopwords))
    
    stopwords_bc = get_stopwords_bc(spark.sparkContext, "./inputs/bahasa_stopwords.txt")
    df = spark.createDataFrame([[1,2],[3,4]], ["a", "b"])
    df.show()


# In[34]:


stopwords_bc.value[:10]


# In[ ]:





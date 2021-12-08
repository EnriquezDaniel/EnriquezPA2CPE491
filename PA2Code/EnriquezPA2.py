from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import PCA
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel

print("Made by Daniel Enriquez for CPE491 (CS442) cloud computing")
print("This code is heavily inspired from: https://piotrszul.github.io/spark-tutorial/notebooks/3.1_ML-Introduction.html")
print("As we were encouraged to write few lines as instructed in class, this was used and modified")

# Creates a spark session
spark = SparkSession.builder.appName("Enriquez PA2").getOrCreate()

# reader
inputDF = spark.read.csv('data/winequality-white.csv',header='true', inferSchema='true', sep=';')

#prints out how the wine data is organized
inputDF.printSchema()
print("Rows: %s" % inputDF.count())

# Creates the feature columns for all columns except the "quality column"
featureColumns = [c for c in inputDF.columns if c != 'quality']

# It then assembles all of the features listed earlier and outputs it to the column
assembler = VectorAssembler(inputCols=featureColumns,
                            outputCol="features")

dataDF = assembler.transform(inputDF)
dataDF.printSchema()

# Here the linear regression is performed
lr = LinearRegression(maxIter=30, regParam=0.3, elasticNetParam=0.3, featuresCol="features", labelCol="quality")
lrModel = lr.fit(dataDF)

predictionsDF = lrModel.transform(dataDF)

# Here it just reads in and evaluates whether the prediction was/wasn't true
evaluator = RegressionEvaluator(
    labelCol='quality', predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictionsDF)
print("Root Mean Squared Error (RMSE) = %g" % rmse)

# Here we determine the average wine quality
avgQuality = inputDF.groupBy().avg('quality').first()[0]
print("Average Quality: ")
print(avgQuality)
zeroModelPredictionsDF = dataDF.select(col('quality'), lit(avgQuality).alias('prediction'))

zeroModelRmse = evaluator.evaluate(zeroModelPredictionsDF)
print("RMSE of 'zero model' = %g" % zeroModelRmse)

# Here is where we split the model from 80% learning 20% testing
(trainingDF, testDF) = inputDF.randomSplit([0.8, 0.2])

# Here is where we pipeline with an assembler and a regression model
pipeline = Pipeline(stages=[assembler, lr])

lrPipelineModel = pipeline.fit(trainingDF)

# Here is where we train/perform testing on the model
traningPredictionsDF = lrPipelineModel.transform(trainingDF)
testPredictionsDF = lrPipelineModel.transform(testDF)

print("RMSE on traning data = %g" % evaluator.evaluate(traningPredictionsDF))

print("RMSE on test data = %g" % evaluator.evaluate(testPredictionsDF))

search_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.0, 0.3, 0.6]) \
    .addGrid(lr.elasticNetParam, [0.4, 0.6, 0.8]).build()

#implements a cross validator to tune the model to select the best sets of parameters
cv = CrossValidator(estimator = pipeline, estimatorParamMaps = search_grid, evaluator = evaluator, numFolds = 3)
cvModel = cv.fit(trainingDF)

cvTestPredictionsDF = cvModel.transform(testDF)
print("RMSE on test data with CV = %g" % evaluator.evaluate(cvTestPredictionsDF))

print("Average metrics from cross validator model")
print(cvModel.avgMetrics)

rf = RandomForestRegressor(featuresCol="features", labelCol="quality", numTrees=100, maxBins=128, maxDepth=20, \
                           minInstancesPerNode=5, seed=33)
rfPipeline = Pipeline(stages=[assembler, rf])

#Trains the random forest model
rfPipelineModel = rfPipeline.fit(trainingDF)


rfTrainingPredictions = rfPipelineModel.transform(trainingDF)
rfTestPredictions = rfPipelineModel.transform(testDF)
print("Random Forest RMSE on traning data = %g" % evaluator.evaluate(rfTrainingPredictions))
print("Random Forest RMSE on test data = %g" % evaluator.evaluate(rfTestPredictions))

rfModel = rfPipelineModel.stages[1]
rfModel.featureImportances

# Here's the program output which writes to a directory names output
rfPipelineModel.write().overwrite().save('output/rf.model')

loadedModel = PipelineModel.load('output/rf.model')
loadedPredictionsDF = loadedModel.transform(testDF)

print("Loaded model RMSE = %g" % evaluator.evaluate(loadedPredictionsDF))


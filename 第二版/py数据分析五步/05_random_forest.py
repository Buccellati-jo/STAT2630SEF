from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, when, size, split
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['HADOOP_HOME'] = os.path.dirname(os.path.abspath(__file__))
os.environ['hadoop.home.dir'] = os.path.dirname(os.path.abspath(__file__))

spark = SparkSession.builder \
    .appName("YouTubeShortVideo_RandomForest") \
    .master("local[*]") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .config("spark.hadoop.util.shutdownhookmanager.enabled", "false") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()
spark.sparkContext.setLogLevel("OFF")

csv_path = r"C:\Users\wu\Desktop\stat小组作业\第二版\py数据分析五步\short_video_dataset_cleaned.csv"
df = spark.read.csv(
    csv_path,
    header=True,
    inferSchema=True,
    multiLine=True
)

df_clean = df.dropDuplicates(["video_id"]) \
    .dropna(subset=["title", "views", "likes", "comments_count", "avg_sentiment"]) \
    .filter(
        (col("views") > 0) & 
        (col("likes") >= 0) & 
        (col("comments_count") > 0)
    ) \
    .filter(
        (col("avg_sentiment") >= -1) & 
        (col("avg_sentiment") <= 1)
    )

df_clean = df_clean.withColumn("publish_hour", hour(col("publish_time")))
df_clean = df_clean.withColumn(
    "time_period",
    when((col("publish_hour") >= 0) & (col("publish_hour") <= 6), 0)
    .when((col("publish_hour") >= 7) & (col("publish_hour") <= 12), 1)
    .when((col("publish_hour") >= 13) & (col("publish_hour") <= 18), 2)
    .otherwise(3)
)
df_clean = df_clean.withColumn("tag_num", size(split(col("title"), "#")) - 1)

assembler = VectorAssembler(
    inputCols=["likes", "comments_count", "avg_sentiment", "tag_num", "time_period"],
    outputCol="features"
)
df_model = assembler.transform(df_clean)

train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestRegressor(featuresCol="features", labelCol="views", numTrees=100, seed=42)
rf_model = rf.fit(train_df)

print("="*80)
print("Random Forest Model Results")
print("="*80)
print("Feature Importances:")
feature_names = ["likes", "comments_count", "avg_sentiment", "tag_num", "time_period"]
for name, importance in zip(feature_names, rf_model.featureImportances):
    print(f"  {name}: {importance:.4f}")

rf_predictions = rf_model.transform(test_df)
evaluator = RegressionEvaluator(labelCol="views", predictionCol="prediction", metricName="r2")
rf_r2 = evaluator.evaluate(rf_predictions)
print(f"Test Set R²: {rf_r2:.4f}")
print("="*80)
print("Random Forest Analysis Completed!")
print("="*80)

spark.stop()
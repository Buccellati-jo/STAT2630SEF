from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, min, max
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['HADOOP_HOME'] = os.path.dirname(os.path.abspath(__file__))
os.environ['hadoop.home.dir'] = os.path.dirname(os.path.abspath(__file__))

spark = SparkSession.builder \
    .appName("YouTubeShortVideo_DescriptiveAnalysis") \
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

print("="*80)
print("Descriptive Statistics of Core Variables")
print("="*80)

df_clean.select(
    mean("views").alias("views_mean"),
    stddev("views").alias("views_std"),
    min("views").alias("views_min"),
    max("views").alias("views_max"),
    mean("likes").alias("likes_mean"),
    stddev("likes").alias("likes_std"),
    min("likes").alias("likes_min"),
    max("likes").alias("likes_max"),
    mean("comments_count").alias("comments_mean"),
    stddev("comments_count").alias("comments_std"),
    min("comments_count").alias("comments_min"),
    max("comments_count").alias("comments_max"),
    mean("avg_sentiment").alias("sentiment_mean"),
    stddev("avg_sentiment").alias("sentiment_std"),
    min("avg_sentiment").alias("sentiment_min"),
    max("avg_sentiment").alias("sentiment_max")
).show(truncate=False)

print("="*80)
print("Descriptive Analysis Completed!")
print("="*80)

spark.stop()
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, corr, hour, when, size, split
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['HADOOP_HOME'] = os.path.dirname(os.path.abspath(__file__))
os.environ['hadoop.home.dir'] = os.path.dirname(os.path.abspath(__file__))

spark = SparkSession.builder \
    .appName("YouTubeShortVideo_CorrelationAnalysis") \
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

print("="*80)
print("Correlation Analysis Results (with views)")
print("="*80)

corr_likes = df_clean.select(corr('views', 'likes')).collect()[0][0]
corr_comments = df_clean.select(corr('views', 'comments_count')).collect()[0][0]
corr_sentiment = df_clean.select(corr('views', 'avg_sentiment')).collect()[0][0]
corr_tags = df_clean.select(corr('views', 'tag_num')).collect()[0][0]
corr_time = df_clean.select(corr('views', 'time_period')).collect()[0][0]

print(f"likes & views: {corr_likes:.4f}")
print(f"comments_count & views: {corr_comments:.4f}")
print(f"avg_sentiment & views: {corr_sentiment:.4f}")
print(f"tag_num & views: {corr_tags:.4f}")
print(f"time_period & views: {corr_time:.4f}")

print("="*80)
print("Correlation Analysis Completed!")
print("="*80)

spark.stop()
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, when, size, split
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['HADOOP_HOME'] = os.path.dirname(os.path.abspath(__file__))
os.environ['hadoop.home.dir'] = os.path.dirname(os.path.abspath(__file__))

spark = SparkSession.builder \
    .appName("YouTubeShortVideo_DataCheck") \
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

print("="*80)
print("1. CSV File Loaded Successfully")
print("="*80)
df.printSchema()
print("="*80)

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

print(f"2. Data Cleaning Completed: Original {df.count()}, Cleaned {df_clean.count()}")
print("="*80)

df_clean = df_clean.withColumn("publish_hour", hour(col("publish_time")))
df_clean = df_clean.withColumn(
    "time_period",
    when((col("publish_hour") >= 0) & (col("publish_hour") <= 6), 0)
    .when((col("publish_hour") >= 7) & (col("publish_hour") <= 12), 1)
    .when((col("publish_hour") >= 13) & (col("publish_hour") <= 18), 2)
    .otherwise(3)
)
df_clean = df_clean.withColumn("tag_num", size(split(col("title"), "#")) - 1)

print("3. Feature Engineering Completed")
print("="*80)
print("4. Processed Data Preview (Top 5 Rows)")
df_clean.select("video_id", "views", "likes", "comments_count", "time_period", "tag_num").show(5, truncate=False)
print("="*80)

spark.stop()
print("\nData Check Pipeline Executed Successfully!")
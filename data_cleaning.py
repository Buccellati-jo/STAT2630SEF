# data_cleaning.py
import pandas as pd
import numpy as np
from pymongo import MongoClient
import config

# 1. 从MongoDB读取原始数据（与CSV结构一致）
client = MongoClient(config.MONGO_URI)
db = client[config.MONGO_DB_NAME]
col_video = db[config.COL_VIDEO]
df_raw = pd.DataFrame(list(col_video.find()))
# 剔除MongoDB自动生成的_id字段，无实际意义
df_raw = df_raw.drop('_id', axis=1, errors='ignore')
client.close()

# 2. 数据清洗核心操作
## 2.1 去重：按video_id去重（唯一标识，避免重复爬取）
df_clean = df_raw.drop_duplicates(subset=['video_id'], keep='first')
## 2.2 去空：删除关键字段空值（标题、播放量、评论、平均情感分不能为空）
df_clean = df_clean.dropna(subset=['title', 'views', 'comments', 'avg_sentiment'])
## 2.3 去异常值：
# - 过滤播放量/点赞量/评论数为0的无效视频（无互动，无建模价值）
# - 过滤平均情感分超出[-1,1]的异常值（情感分析计算错误）
df_clean = df_clean[
    (df_clean['views'] > 0) &
    (df_clean['likes'] >= 0) &
    (df_clean['comments_count'] > 0) &
    (df_clean['avg_sentiment'] >= -1) &
    (df_clean['avg_sentiment'] <= 1)
]
# 重置索引
df_clean = df_clean.reset_index(drop=True)

# 3. 保存清洗后的数据
df_clean.to_csv('short_video_dataset_cleaned.csv', index=False, encoding='utf-8-sig')
print(f"数据清洗完成！原始数据{len(df_raw)}条，清洗后{len(df_clean)}条，已保存为short_video_dataset_cleaned.csv")
print(f"清洗后数据字段：{list(df_clean.columns)}")
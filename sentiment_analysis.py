# sentiment_analysis.py
import nltk
import pandas as pd
from pymongo import MongoClient
import config
from nltk.sentiment import SentimentIntensityAnalyzer

# 下载VADER词典（首次运行需执行，后续注释）
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# 1. 连接MongoDB，读取评论数据
client = MongoClient(config.MONGO_URI)
db = client[config.MONGO_DB_NAME]
col_comment = db[config.COL_COMMENT]
col_video = db[config.COL_VIDEO]

# 2. 计算每条评论的情感评分，更新MongoDB
comments = list(col_comment.find({}))
for comment in comments:
    comment_text = comment['comment']
    # 计算复合情感分（VADER核心指标，范围[-1,1]）
    sentiment = sia.polarity_scores(comment_text)['compound']
    # 更新MongoDB评论集合的情感分
    col_comment.update_one(
        {'_id': comment['_id']},
        {'$set': {'sentiment_score': sentiment}}
    )

# 3. 重新聚合评论，计算每个视频的平均情感分，更新视频集合和CSV
df_comment = pd.DataFrame(list(col_comment.find()))
df_comment_agg = df_comment.groupby('video_id').agg(
    comments=('comment', lambda x: ' | '.join(x)),
    avg_sentiment=('sentiment_score', 'mean')
).reset_index()
df_video = pd.DataFrame(list(col_video.find()))
df_raw = pd.merge(df_video, df_comment_agg, on='video_id', how='left')

# 4. 更新MongoDB视频集合的平均情感分
for _, row in df_raw.iterrows():
    col_video.update_one(
        {'video_id': row['video_id']},
        {'$set': {'avg_sentiment': row['avg_sentiment'], 'comments': row['comments']}}
    )

# 5. 保存最终原始数据集CSV（含情感分，与原A角色产出一致）
df_raw.to_csv('short_video_dataset.csv', index=False, encoding='utf-8-sig')
client.close()
print("情感评分计算完成！已更新MongoDB和short_video_dataset.csv，包含平均情感分和单条评论情感分")
# crawl_youtube.py
from googleapiclient.discovery import build
from pymongo import MongoClient
import pandas as pd
import config

# 1. 初始化YouTube API和MongoDB连接
youtube = build('youtube', 'v3', developerKey=config.YOUTUBE_API_KEY)
client = MongoClient(config.MONGO_URI)
db = client[config.MONGO_DB_NAME]
col_video = db[config.COL_VIDEO]
col_comment = db[config.COL_COMMENT]
# 清空集合（避免重复数据，首次运行可保留）
col_video.delete_many({})
col_comment.delete_many({})

# 2. 爬取短视频基础信息（按shorts标签筛选，可自定义筛选条件）
video_list = []
next_page_token = None
while len(video_list) < config.CRAWL_VIDEO_NUM:
    request = youtube.search().list(
        q="shorts",  # 搜索关键词，可替换为其他标签（如comedy、food）
        part="snippet",
        type="video",
        maxResults=50,  # 每次最多爬50条，API限制
        pageToken=next_page_token
    )
    response = request.execute()
    # 遍历获取视频基础信息，补充统计数据（播放、点赞、评论）
    for item in response['items']:
        if len(video_list) >= config.CRAWL_VIDEO_NUM:
            break
        video_id = item['id']['videoId']
        # 获取视频统计数据
        stats_request = youtube.videos().list(part="statistics", id=video_id)
        stats_resp = stats_request.execute()
        stats = stats_resp['items'][0]['statistics']
        # 构造视频数据字典（处理缺失值，避免键不存在报错）
        video_data = {
            "video_id": video_id,
            "title": item['snippet']['title'],
            "channel": item['snippet']['channelTitle'],
            "publish_time": item['snippet']['publishedAt'],
            "views": int(stats.get('viewCount', 0)),
            "likes": int(stats.get('likeCount', 0)),
            "comments_count": int(stats.get('commentCount', 0))
        }
        video_list.append(video_data)
        # 写入MongoDB视频集合
        col_video.insert_one(video_data)
    next_page_token = response.get('nextPageToken')

# 3. 爬取每个视频的评论，最多100条/视频
comment_list = []
for video in video_list:
    video_id = video['video_id']
    next_comment_token = None
    comment_num = 0
    while comment_num < config.CRAWL_COMMENT_NUM:
        try:
            comment_request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_comment_token
            )
            comment_resp = comment_request.execute()
            for item in comment_resp['items']:
                if comment_num >= config.CRAWL_COMMENT_NUM:
                    break
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comment_data = {
                    "video_id": video_id,
                    "comment": comment,
                    "sentiment_score": 0.0  # 先占位，后续计算情感评分后更新
                }
                comment_list.append(comment_data)
                # 写入MongoDB评论集合
                col_comment.insert_one(comment_data)
                comment_num += 1
            next_comment_token = comment_resp.get('nextPageToken')
            if not next_comment_token:
                break
        except Exception as e:
            print(f"视频{video_id}爬取评论失败：{e}")
            break

# 4. 初步生成CSV文件（未计算情感分，后续更新）
# 合并视频和评论（聚合评论为字符串，平均情感分先占位）
df_video = pd.DataFrame(video_list)
df_comment = pd.DataFrame(comment_list)
df_comment_agg = df_comment.groupby('video_id').agg(
    comments=('comment', lambda x: ' | '.join(x)),
    avg_sentiment=('sentiment_score', 'mean')
).reset_index()
df_raw = pd.merge(df_video, df_comment_agg, on='video_id', how='left')
# 保存原始CSV（未计算情感分）
df_raw.to_csv('short_video_dataset_raw.csv', index=False, encoding='utf-8-sig')
print(f"爬取完成！共{len(video_list)}个视频，{len(comment_list)}条评论，已存入MongoDB和short_video_dataset_raw.csv")
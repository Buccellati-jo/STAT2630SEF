# feature_engineering.py
import pandas as pd
import re
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import config

# 1. 读取清洗后的数据
df_clean = pd.read_csv('short_video_dataset_cleaned.csv', encoding='utf-8-sig')
# 初始化特征表
df_feature = df_clean.copy()

# 2. 特征工程1：发布时间转时段（凌晨/早/中/晚，分类特征）
## 2.1 解析publish_time为datetime格式，提取小时
df_feature['publish_time'] = pd.to_datetime(df_feature['publish_time'])
df_feature['publish_hour'] = df_feature['publish_time'].dt.hour
## 2.2 分时段：凌晨(0-6)/早上(7-12)/中午(13-18)/晚上(19-23)
def get_time_period(hour):
    if 0 <= hour <= 6:
        return '凌晨'
    elif 7 <= hour <= 12:
        return '早上'
    elif 13 <= hour <= 18:
        return '中午'
    else:
        return '晚上'
df_feature['time_period'] = df_feature['publish_hour'].apply(get_time_period)
# 标签编码：将分类时段转为数值（0/1/2/3），适配建模
le_period = LabelEncoder()
df_feature['time_period_enc'] = le_period.fit_transform(df_feature['time_period'])

# 3. 特征工程2：提取标题关键词+标签转特征（文本特征，核心高分点）
## 3.1 提取标题中的标签（#后内容），生成标签列表/标签数量
def extract_tags(title):
    tags = re.findall(r'#(\w+)', title)  # 匹配#后的字母/数字
    return tags if tags else ['no_tag']
df_feature['tags'] = df_feature['title'].apply(extract_tags)
df_feature['tag_num'] = df_feature['tags'].apply(len)  # 衍生特征：标签数量
## 3.2 标签转独热编码（前20个高频标签，避免维度爆炸）
all_tags = [tag for tags in df_feature['tags'] for tag in tags]
top20_tags = pd.Series(all_tags).value_counts().head(20).index.tolist()
for tag in top20_tags:
    df_feature[f'tag_{tag}'] = df_feature['tags'].apply(lambda x: 1 if tag in x else 0)
## 3.3 提取标题关键词（去除标签/停用词，生成关键词特征）
# 预处理标题：去除标签、特殊符号，保留纯文本
def clean_title(title):
    title = re.sub(r'#\w+', '', title)  # 去除标签
    title = re.sub(r'[^\w\s]', '', title)  # 去除特殊符号
    return title.strip().lower()
df_feature['clean_title'] = df_feature['title'].apply(clean_title)
# 词袋模型提取标题关键词（前15个高频关键词，生成数值特征）
cv = CountVectorizer(max_features=15, stop_words='english')  # 停用词过滤
title_keyword = cv.fit_transform(df_feature['clean_title']).toarray()
keyword_cols = [f'keyword_{word}' for word in cv.get_feature_names_out()]
df_keyword = pd.DataFrame(title_keyword, columns=keyword_cols)
df_feature = pd.concat([df_feature, df_keyword], axis=1)

# 4. 特征工程3：时长分段（短视频无直接时长，两种方案）
## 方案1：API补充爬取时长（精准，推荐）：在crawl_youtube.py中补充爬取duration字段，再分箱
## 方案2：互动率推导（无API，适配作业）：按「点赞率=点赞/播放」分三段，替代时长分段
df_feature['like_rate'] = df_feature['likes'] / df_feature['views']  # 点赞率（衍生特征）
df_feature['comment_rate'] = df_feature['comments_count'] / df_feature['views']  # 评论率（衍生特征）
# 按点赞率分三段：低互动(0-0.01)/中互动(0.01-0.05)/高互动(>0.05)，对应短视频时长（短/中/长）
def get_duration_segment(rate):
    if rate < 0.01:
        return '短时长'
    elif 0.01 <= rate <= 0.05:
        return '中时长'
    else:
        return '长时长'
df_feature['duration_segment'] = df_feature['like_rate'].apply(get_duration_segment)
# 标签编码：时长分段转数值
le_duration = LabelEncoder()
df_feature['duration_segment_enc'] = le_duration.fit_transform(df_feature['duration_segment'])

# 5. 特征工程4：数值特征标准化（播放量/点赞量/评论数等，适配建模）
scaler = MinMaxScaler()  # 归一化到[0,1]
num_feat = ['views', 'likes', 'comments_count', 'like_rate', 'comment_rate', 'tag_num']
df_feature[num_feat] = scaler.fit_transform(df_feature[num_feat])

# 6. 筛选建模特征（剔除无用字段，保留特征字段）
useless_cols = ['publish_time', 'publish_hour', 'tags', 'clean_title', 'time_period', 'duration_segment', 'comments']
model_feat_cols = [col for col in df_feature.columns if col not in useless_cols]
df_model = df_feature[model_feat_cols]

# 7. 保存最终建模表+特征说明
## 7.1 保存建模CSV（可直接导入机器学习模型）
df_model.to_csv('model_dataset.csv', index=False, encoding='utf-8-sig')
## 7.2 生成特征说明文档（txt，给建模组使用，作业必备）
feature_desc = []
feature_desc.append("=== YouTube短视频建模特征说明 ===")
feature_desc.append(f"特征总数：{len(model_feat_cols)}，数据量：{len(df_model)}")
feature_desc.append("\n1. 基础标识特征：")
feature_desc.append("video_id：视频唯一标识（非建模特征，用于溯源）")
feature_desc.append("channel：频道名（原始分类特征，可进一步编码）")
feature_desc.append("\n2. 标准化数值特征（[0,1]）：")
for col in num_feat:
    feature_desc.append(f"{col}：{col.replace('_', ' ')}，归一化后数值")
feature_desc.append("\n3. 分类特征（标签编码后，0/1/2/3）：")
feature_desc.append("time_period_enc：发布时段（0=凌晨，1=早上，2=中午，3=晚上）")
feature_desc.append("duration_segment_enc：时长分段（0=短时长，1=中时长，2=长时长）")
feature_desc.append("\n4. 标签独热特征（1=包含该标签，0=不包含）：")
feature_desc.append(f"{', '.join([col for col in df_feature.columns if col.startswith('tag_')])}")
feature_desc.append("\n5. 标题关键词特征（词袋模型，数值为关键词出现次数）：")
feature_desc.append(f"{', '.join(keyword_cols)}")
feature_desc.append("\n6. 情感特征：")
feature_desc.append("avg_sentiment：视频平均情感分（[-1,1]，负数=负面，正数=正面）")
# 写入特征说明文件
with open('feature_description.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(feature_desc))

# 8. 将建模特征表写入MongoDB（可选，工程化需求）
from pymongo import MongoClient
client = MongoClient(config.MONGO_URI)
db = client[config.MONGO_DB_NAME]
col_model = db[config.COL_MODEL]
col_model.delete_many({})
col_model.insert_many(df_model.to_dict('records'))
client.close()

print("特征工程完成！已生成最终建模表model_dataset.csv和特征说明feature_description.txt")
print(f"建模特征共{len(model_feat_cols)}个，包含数值/分类/文本/情感多维度特征")
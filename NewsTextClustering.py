# 导入库
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # 基于TF-IDF的词频转向量库
from sklearn.cluster import KMeans
import jieba.posseg as pseg
import newspaper
import requests  # 导入爬虫库
from bs4 import BeautifulSoup
import datetime as dt

today = dt.datetime.today().strftime("%Y-%m/%d")  # 过去今天时间


# 中文分词
def jieba_cut(comment):
    word_list = []  # 建立空列表用于存储分词结果
    seg_list = pseg.cut(comment)  # 精确模式分词[默认模式]
    for word in seg_list:
        if word.flag in ['ns', 'n', 'vn', 'v', 'nr']:  # 选择属性
            word_list.append(word.word)  # 分词追加到列表
    return word_list


# 获取新闻内容
def get_news():
    # headers = {"Host": "www.bjnews.com.cn",
    #            "Referer": "http://www.bjnews.com.cn/",
    #            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36",
    #            "X-Requested-With": "XMLHttpRequest"
    #            }
    # s_url = 'http://www.bjnews.com.cn/'
    s_url = 'http://www.xinhuanet.com/world/'
    response = requests.get(s_url)
    html = response.content.decode('utf-8')
    soup = BeautifulSoup(html, 'lxml')
    all_a = soup.find_all('a')
    comment_list = []  # 建立空列表用于存储分词结果

    for a in all_a:
        try:
            url = a['href']
            if (s_url in url) & (today in url):
            # if (s_url in url):
                article = newspaper.Article(url, language='zh')
                # 下载文章
                article.download()
                # 解析文章
                article.parse()
                # 对文章进行nlp处理
                article.nlp()
                # 获取文章的内容
                article_words = "".join(article.keywords)
                comment_list.append(article_words)
        except:
            pass
    return comment_list


comment_list = get_news()
# print(comment_list)
# word to vector
# 加载停用词
stop_words = [line.strip() for line in open('stopword.txt', encoding='utf-8').readlines()]
vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=jieba_cut, use_idf=True)  # 创建词向量模型
X = vectorizer.fit_transform(comment_list)  # 将评论关键字列表转换为词向量空间模型
# K均值聚类
model_kmeans = KMeans(n_clusters=3)  # 创建聚类模型对象
model_kmeans.fit(X)  # 训练模型
# 聚类结果汇总
cluster_labels = model_kmeans.labels_  # 聚类标签结果
word_vectors = vectorizer.get_feature_names()  # 词向量
word_values = X.toarray()  # 向量值
comment_matrix = np.hstack((word_values, cluster_labels.reshape(word_values.
                                                                shape[0], 1)))  # 将向量值和标签值合并为新的矩阵
word_vectors.append('cluster_labels')  # 将新的聚类标签列表追加到词向量后面
comment_pd = pd.DataFrame(comment_matrix, columns=word_vectors)  # 创建包含词向量和聚类标签的数据框
comment_pd.to_csv('comment.csv')
print(comment_pd.head(1))  # 打印输出数据框第1条数据
# 聚类结果分析
comment_cluster1 = comment_pd[comment_pd['cluster_labels'] == 1].drop('cluster_labels', axis=1)  # 选择聚类标签值为1的数据，并删除最后一列
word_importance = np.sum(comment_cluster1, axis=0)  # 按照词向量做汇总统计
print(word_importance.sort_values(ascending=False)[:5])  # 按汇总统计的值做逆序排序并打印输出前5个词

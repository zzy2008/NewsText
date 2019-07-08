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
import os
from os import listdir
import sys
import GetKeyWord
from gensim import corpora
from gensim import models
from gensim.models import word2vec
import logging
from gensim import utils
import itertools

today = dt.datetime.today().strftime("%Y/%m/%d")  # 过去今天时间


class LineSentence(object):
    """Iterate over a file that contains sentences: one line = one sentence.
    Words must be already preprocessed and separated by whitespace.

    """

    def __init__(self, source, max_sentence_length=10000, limit=None):
        """

        Parameters
        ----------
        source : string or a file-like object
            Path to the file on disk, or an already-open file object (must support `seek(0)`).
        limit : int or None
            Clip the file to the first `limit` lines. Do no clipping if `limit is None` (the default).

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> sentences = LineSentence(datapath('lee_background.cor'))
            >>> for sentence in sentences:
            ...     pass

        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = utils.to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield line[i: i + self.max_sentence_length]
                    i += self.max_sentence_length
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source, mode="r") as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i: i + self.max_sentence_length]
                        i += self.max_sentence_length


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
    # headers = {"Host": "www.xinhuanet.com",
    #            "Referer": "http://www.xinhuanet.com/world/",
    #            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36",
    #            "X-Requested-With": "XMLHttpRequest"
    #            }
    s_url = 'http://www.bjnews.com.cn/world/'
    # s_url = 'http://www.xinhuanet.com/world/'
    response = requests.get(s_url)
    html = response.content.decode('utf-8')
    soup = BeautifulSoup(html, 'lxml')
    all_a = soup.find_all('a')
    article_list = []  # 建立空列表用于文章结果
    url_list = []

    for a in all_a:
        try:
            url = a['href']
            if (s_url in url) & (today in url) & (url not in url_list):
                # if (s_url in url) &(url not in url_list):
                url_list.append(url)
                article = newspaper.Article(url, language='zh')
                # 下载文章
                article.download()
                # 解析文章
                article.parse()
                # 对文章进行nlp处理
                article.nlp()
                # 获取文章的内容
                article_words = "".join(article.keywords)
                article_list.append(article_words)
        except:
            pass
    return article_list, url_list


def cut_article(cut_model, article_list):
    cut_article_list = []  # 用于存储分词后的文章列表，每个元素为一篇文章，每篇文章含有分好的若干词
    for article in article_list:
        cut_article_list.append(jieba_cut(article))
    return cut_article_list


def save_cut_article_list(cut_article_list):
    temp = sys.stdout
    for idx, article in enumerate(cut_article_list):
        with open(r"article_list/article_{}.txt".format(idx + 1), "w") as f:
            sys.stdout = f
            for word in article:
                print(word, end=' ')
    with open(r"article_list/article_all.txt", "w") as f:
        sys.stdout = f
        for article in cut_article_list:
            for word in article:
                print(word, end=' ')
            print('\r', end='')
    sys.stdout = temp





def gensim_analysis(cut_model, cut_article_list, n_clusters, rank):
    # stop_words = [line.strip() for line in open('stopword.txt', encoding='utf-8').readlines()]
    # vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=cut_model, use_idf=True)  # 创建词向量模型
    # X = vectorizer.fit_transform(article_list)
    # print(X)

    # docLabels = []
    # docLabels = [file for file in listdir("article_list") if file.endswith('.txt')]
    # data = []
    # for doc in docLabels:
    #     for doc in docLabels:
    #         data.append(open('article_list/' + doc, 'r'))

    # dictionary = corpora.Dictionary(cut_article_list)
    # corpus = [dictionary.doc2bow(article) for article in cut_article_list]
    # print(corpus)
    # tfidf = models.TfidfModel(corpus)
    # doc_bow = [(0, 1), (1, 1), (4, 5)]
    # print(tfidf[doc_bow])

    # logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    # sentences = []
    # sentences = LineSentence(r'article_list/article_all.txt')
    # model = word2vec.Word2Vec(sentences, size=100, hs=1, min_count=1, window=3)
    # print("-----------------------------------")
    # try:
    #     sim1 = model.similarity(u'特朗普', u'经济')
    #     sim2 = model.similarity(u'伊朗', u'核')
    # except KeyError:
    #     sim1 = 0
    #     sim2 = 0
    # print('sim1: ', sim1)
    # print('sim2: ', sim2)
    #
    # print('-----------------------------------')
    # # 与某个词（李达康）最相近的3个字的词
    # print(u'与特朗普最相近的3个字的词')
    # req_count = 5
    # for key in model.similar_by_word(u'特朗普', topn=100):
    #     if len(key[0]) == 3:
    #         req_count -= 1
    #         print(key[0], key[1])
    #         if req_count == 0:
    #             break
    #
    # print('-----------------------------------')
    # # 计算某个词(侯亮平)的相关列表
    # try:
    #     sim3 = model.most_similar(u'特朗普', topn=20)
    #     print(u'和 特朗普 与相关的词有：\n')
    #     for key in sim3:
    #         print(key[0], key[1])
    # except:
    #     print(' error')




def kmeans_analysis(cut_model, article_list, n_clusters, rank):  # 分词模型，单词列表，k值，输出高频位数
    # word to vector
    # 加载停用词
    stop_words = [line.strip() for line in open('stopword.txt', encoding='utf-8').readlines()]
    vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=cut_model, use_idf=True)  # 创建词向量模型
    X = vectorizer.fit_transform(article_list)  # 将评论关键字列表转换为词向量空间模型
    # K均值聚类
    model_kmeans = KMeans(n_clusters=n_clusters)  # 创建聚类模型对象
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
    print("--------------------词向量和聚类标签的Frame--------------------")
    print(comment_pd.head())  # 打印输出数据框5条数据

    # 聚类结果分析
    temp = sys.stdout
    with open("ClusteringResult .txt", "w") as f:
        sys.stdout = f
        print("|------------------------------|")
        print("|----kMeansClusteringResult----|")
        print("|------------------------------|")
        for i in range(3):
            comment_cluster = comment_pd[comment_pd['cluster_labels'] == i].drop('cluster_labels',
                                                                                 axis=1)  # 选择聚类标签值为1的数据，并删除最后一列
            word_importance = np.sum(comment_cluster, axis=0)  # 按照词向量做汇总统计
            print("第{}类的前{}位关键高频词：".format(i + 1, rank))
            print("--------------------------------")
            print(word_importance.sort_values(ascending=False)[:rank])  # 按汇总统计的值做逆序排序并打印输出前5个词
            print("--------------------------------")

def save_url_list(url_list):
    temp = sys.stdout  # 记录当前输出指向，默认是consle
    with open("URL_List.txt", "w") as f:
        sys.stdout = f
        print("------------URL List------------")
        for url in url_list:
            print(url)
        print("------------URL List------------")

        sys.stdout = temp
def save_key_word(article_list):
    temp = sys.stdout
    with open("KeyWord.txt", "w") as f:
        sys.stdout = f  # 输出指向txt文件
        print("|-------------------------------|")
        print("|----articles keyword Record----|")
        print("|-------------------------------|")
        keywords = ""
        newscount = len(article_list)
        for idx, article_word in enumerate(article_list):
            keyword = GetKeyWord.get_key_words(article_word)
            print("News No.{}".format(idx + 1))
            print("---------------------------------")
            print(keyword)
            print("---------------------------------")
        print("Number of articles: {}".format(newscount))
        sys.stdout = temp  # 输出重定向回consle
if __name__ == '__main__':
    #从新闻网站爬取数据
    article_list, url_list = get_news()
    #存储爬取的网站链接
    save_url_list(url_list)
    #存储每篇文章的关键词
    save_key_word(article_list)
    #对文本进行分词
    cut_article_list = cut_article(jieba_cut, article_list)
    #存储分词后的文本
    save_cut_article_list(cut_article_list)

    gensim_analysis(jieba_cut, cut_article_list, n_clusters=3, rank=5)
    kmeans_analysis(jieba_cut, article_list, n_clusters=3, rank=5)

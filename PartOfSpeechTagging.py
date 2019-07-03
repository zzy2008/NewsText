import newspaper
import pandas as pd
import jieba.posseg as pseg

# 加载停用词
stopWords = [line.strip() for line in open('stopword.txt', encoding='utf-8').readlines()]

# 获取文章 银保监会出台新政为例
# article = newspaper.Article('https://finance.sina.com.cn/money/bank/bank_hydt/2019-02-25/doc-ihsxncvf7656807.shtml',
#                             language='zh')
# article = newspaper.Article('http://www.bjnews.com.cn/world/2019/06/30/597429.html',
#                             language='zh')
article = newspaper.Article('http://www.bjnews.com.cn/world/2019/06/30/597407.html',
                            language='zh')
article.download()
# 解析文章
article.parse()
# 对文章进行nlp处理
article.nlp()
# nlp处理后的文章拼接
article_words = "".join(article.keywords)

seg_list_exact = pseg.cut(article_words)  # 精确模式分词[默认模式]

words_list = []  # 空列表用于存储分词和词性分类

for word in seg_list_exact:  # 循环得到每个分词
    if word not in stopWords:  # 如果不在去除词库中
        words_list.append((word.word, word.flag))  # 将分词和词性分类追加到列表

words_pd = pd.DataFrame(words_list, columns=['word', 'type'])  # 创建结果数据框
print(words_pd.head())  # 展示
# print(words_pd)  # 展示

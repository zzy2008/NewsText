# 导入库
import jieba.analyse  # 导入关键字提取库
import pandas as pd  # 导入pandas
import newspaper

# 读取文本数据
# 获取文章 银保监会出台新政为例
article = newspaper.Article('http://www.bjnews.com.cn/news/2019/07/02/598100.html', language='zh')
# 下载文章
article.download()
# 解析文章
article.parse()
# 对文章进行nlp处理
article.nlp()
# nlp处理后的文章拼接
string_data = "".join(article.keywords)


# 关键字提取
def get_key_words(string_data, how=''):
    # topK：提取的关键字数量，不指定则提取全部；
    # withWeight：设置为True指定输出词对应的IF-IDF权重
    if how == 'textrank':
        # 使用TextRank 算法
        tags_pairs = jieba.analyse.textrank(string_data, topK=5, withWeight=True)  # 提取关键字标签
    else:
        # 使用TF-IDF 算法
        tags_pairs = jieba.analyse.extract_tags(string_data, topK=5, withWeight=True)  # 提取关键字标签
    tags_list = []  # 空列表用来存储拆分后的三个值
    for i in tags_pairs:  # 打印标签、分组和TF-IDF权重
        tags_list.append((i[0], i[1]))  # 拆分三个字段值
    tags_pd = pd.DataFrame(tags_list, columns=['word', 'weight'])  # 创建数据框
    return tags_pd


keywords = get_key_words(string_data)
print("#####################TF-IDF####################")
print(keywords)

keywords_tr = get_key_words(string_data, how='textrank')
print("#####################textrank####################")
print(keywords_tr)

# coding=gbk
import pickle


def load_word2vec_model(file_name):
    pickleFilePath = 'model/{}_model.pickle'.format(file_name)
    with open(pickleFilePath, 'rb') as f:
        word2vec_model = pickle.load(f)
    return word2vec_model


def word2vec_similarity(word2vec_model, word1, word2):
    print("-----------------------------------")
    # 相似度计算
    try:
        sim = word2vec_model.wv.similarity(word1, word2)
    except KeyError:
        sim = 0
    print('{}和{}的相似度:{}'.format(word1, word2, sim))


def word2vec_similar_by_word(word2vec_model, word):
    print('-----------------------------------')
    # 与某个词（特朗普）最相近的词
    print(u'与{}最相近的词'.format(word))
    req_count = 5
    for key in word2vec_model.wv.similar_by_word(word, topn=10):
        print(key[0], key[1])


def word2vec_most_similar(word2vec_model, word):
    print('-----------------------------------')
    # 计算某个词(特朗普)的相关列表
    try:
        sim = word2vec_model.wv.most_similar(word, topn=20)
        print(u'和{}相关的词有：\n'.format(word))
        for key in sim:
            print(key[0], key[1])
    except:
        print(' error')


if __name__ == '__main__':
    file_name = 'sina'
    word2vec_model = load_word2vec_model(file_name=file_name)
    word = '伊朗'
    word2vec_most_similar(word2vec_model, word)
    word2vec_similar_by_word(word2vec_model, word)
    word2vec_similarity(word2vec_model, word1=word, word2='伊朗')
    word2vec_similarity(word2vec_model, word1='俄罗斯', word2='伊朗')
    word2vec_similarity(word2vec_model, word1='日本', word2='韩国')

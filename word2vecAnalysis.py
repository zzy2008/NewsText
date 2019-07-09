# coding=gbk
import pickle


def load_word2vec_model(file_name):
    pickleFilePath = 'model/{}_model.pickle'.format(file_name)
    with open(pickleFilePath, 'rb') as f:
        word2vec_model = pickle.load(f)
    return word2vec_model


def word2vec_similarity(word2vec_model, word1, word2):
    print("-----------------------------------")
    # ���ƶȼ���
    try:
        sim = word2vec_model.wv.similarity(word1, word2)
    except KeyError:
        sim = 0
    print('{}��{}�����ƶ�:{}'.format(word1, word2, sim))


def word2vec_similar_by_word(word2vec_model, word):
    print('-----------------------------------')
    # ��ĳ���ʣ������գ�������Ĵ�
    print(u'��{}������Ĵ�'.format(word))
    req_count = 5
    for key in word2vec_model.wv.similar_by_word(word, topn=10):
        print(key[0], key[1])


def word2vec_most_similar(word2vec_model, word):
    print('-----------------------------------')
    # ����ĳ����(������)������б�
    try:
        sim = word2vec_model.wv.most_similar(word, topn=20)
        print(u'��{}��صĴ��У�\n'.format(word))
        for key in sim:
            print(key[0], key[1])
    except:
        print(' error')


if __name__ == '__main__':
    file_name = 'sina'
    word2vec_model = load_word2vec_model(file_name=file_name)
    word = '����'
    word2vec_most_similar(word2vec_model, word)
    word2vec_similar_by_word(word2vec_model, word)
    word2vec_similarity(word2vec_model, word1=word, word2='����')
    word2vec_similarity(word2vec_model, word1='����˹', word2='����')
    word2vec_similarity(word2vec_model, word1='�ձ�', word2='����')

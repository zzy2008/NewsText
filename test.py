stop_words = [line.strip() for line in open('stopword.txt', encoding='utf-8').readlines()]
for idx, word in enumerate(stop_words):
    print(word,end=' ')
    if idx % 20 ==0:
        print("\n")
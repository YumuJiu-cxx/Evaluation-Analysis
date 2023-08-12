import re
import jieba
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def remove_punctuation(line):
    """删除除字母、数字、汉字以外的所有符号"""
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)

    return line


def stopwordslist(filepath):
    """删除停用词"""
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]

    return stopwords


def predict_classify(text, tokenizer, model):
    stopwords = stopwordslist("Data/chineseStopWords.txt")  # 加载停用词
    idx_to_label = np.load('Data/idx_to_label.npy', allow_pickle=True).item()  # 读取词库对应表
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in list(jieba.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=250)
    pred = model.predict(padded)
    classify = pred.argmax(axis=1)[0]
    classify = idx_to_label[classify]

    return classify


def predict_scoring(text, tokenizer, model):
    stopwords = stopwordslist("Data/chineseStopWords.txt")  # 加载停用词
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in list(jieba.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=250)
    scoring = model.predict(padded)[0][0]

    return scoring

from tensorflow.keras import *
import numpy as np
import re
import pickle
import jieba

from lib import *
from keras.preprocessing.sequence import pad_sequences

model = models.load_model('Model/comment_scoring_model.h5')

# 加载tokenizer
with open('Tokenizer/tokenizer_scoring.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 加载停用词
stopwords = stopwordslist("Data/chineseStopWords.txt")


def predict(text):
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in list(jieba.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=250)
    scoring = model.predict(padded)[0][0]

    return scoring


pred = predict('一般')
print(pred)

from tensorflow.keras import *
import numpy as np
import re
import pickle
import jieba

from lib import *
from keras.utils import pad_sequences

model = models.load_model('Model/comment-classification_model.h5')
idx_to_label = np.load('idx_to_label/idx_to_label.npy', allow_pickle=True).item()  # 读取词库对应表

# 加载tokenizer
with open('Tokenizer/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 加载停用词
stopwords = stopwordslist("Data/chineseStopWords.txt")


def predict(text):
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in list(jieba.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=250)
    pred = model.predict(padded)
    classify = pred.argmax(axis=1)[0]

    return classify


pred = predict('垃圾')
pred = idx_to_label[pred]
print(pred)

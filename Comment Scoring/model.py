import pandas as pd
import numpy as np
import re
import os
import jieba
import pickle
from lib import *
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('Data/dataframe.csv', encoding='utf-8')
print(df, '\n')

df = df[['label', 'review']]
print("数据总量:", len(df))
print(df.sample(10), '\n')

# 查看空值数量
print("在 label 列中总共有 %d 个空值." % df['label'].isnull().sum())
print("在 review 列中总共有 %d 个空值." % df['review'].isnull().sum(), '\n')

# 取出非空值的列
df = df[pd.notnull(df['review'])]

# 统计各类别的数据量
d = {'label': df['label'].value_counts().index, 'count': df['label'].value_counts()}
df_label = pd.DataFrame(data=d).reset_index(drop=True)
print('各类别的数据量:')
print(df_label, '\n')

# 加载停用词
stopwords = stopwordslist("Data/chineseStopWords.txt")

# 删除除字母、数字、汉字以外的所有符号
df['clean_review'] = df['review'].apply(remove_punctuation)
# 分词，并过滤停用词
df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))
print(df, '\n')

"""
  LSTM建模
"""
# 定义常量
max_nb_words = 50000  # 设置最频繁使用的50000个词
max_sequence_length = 250  # 设置每条df['clean_review']最大的长度
embedding_dim = 100  # 设置Embedding层的维度

tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['cut_review'].values)
word_index = tokenizer.word_index
print('共有 %s 个不相同的词语.' % len(word_index), '\n')

# 保存tokenizer
with open('Tokenizer/tokenizer_scoring.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X = tokenizer.texts_to_sequences(df['cut_review'].values)
X = pad_sequences(X, maxlen=max_sequence_length)  # 填充X,让X的各个列的长度统一

# 多类标签的onehot展开
Y = df['label']

print(X.shape)
print(Y.shape, '\n')

# 拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
print('X_train维度为:', X_train.shape, '\nY_train维度为:', Y_train.shape)
print('X_test维度为:', X_test.shape, '\nY_test维度为:', Y_test.shape, '\n')

# 定义模型
model = Sequential()
model.add(keras.layers.Embedding(max_nb_words, embedding_dim, input_length=X.shape[1]))
model.add(keras.layers.SpatialDropout1D(0.2))
model.add(keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 加载model参数，继续训练
input_dir = "checkpoints"
model.load_weights(tf.train.latest_checkpoint(input_dir))  # 加载model的权重

"""
  callback模块-checkpoints
"""
output_dir = "checkpoints_2"
if not os.path.exists(output_dir):  # 如果没有此文件夹，则新建
    os.mkdir(output_dir)
checkpoint_prefix = os.path.join(output_dir, 'ckpt_{epoch}')  # 连接两个路径名组件，./text_generation_checkpoints/ckpt_{epoch}
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)  # 保存模型权重用于回调

"""
  训练模型
"""
epochs = 5
batch_size = 64

# 训练模型
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint_callback])
# 自动找到最近保存的变量文件
new_checkpoint = tf.train.latest_checkpoint(output_dir)

# 保存模型
model.save('Model/comment-scoring_model.h5')

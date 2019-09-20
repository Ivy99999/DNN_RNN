
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ivynie

import os

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
import numpy as np
import tensorflow as tf
warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)

file_path = "data"

# 读取用户数据
# 用户ID、性别、年龄、职业ID和邮编
users_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
users = pd.read_table(os.path.join(file_path, "ml-1m/users.dat"), names=users_title, sep="::")
# print(users.head())
movies_title = ['MovieID', 'Title', 'Genres']
movies = pd.read_table(os.path.join(file_path, "ml-1m/movies.dat"), names=movies_title, sep="::")

# 评分数据
# 用户ID、电影ID、评分和时间戳
ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
ratings = pd.read_table(os.path.join(file_path, "ml-1m/ratings.dat"), names=ratings_title, sep='::')

# 进行浅拷贝
movies_source = movies.copy()
# 构建电影id和电影名称的字典表
id2title = pd.Series(movies_source["Title"].values, index=movies_source["MovieID"]).to_dict()




def rec_similar_style(movie_id, topk=20):
    """推荐相似电影"""
    # 从txt文件读取数据
    movie_vec = tf.constant(np.loadtxt(os.path.join(file_path, "data_vec_rnn", "movie_vec.txt"), dtype=np.float32))
    # 对向量normalize
    norm = tf.sqrt(tf.reduce_sum(tf.square(movie_vec), axis=-1, keepdims=True))
    norm_movie_vec = movie_vec / norm
    id_vec = tf.nn.embedding_lookup(norm_movie_vec, np.array([[movie_id]]))
    id_vec = tf.reshape(id_vec, [-1, 256])
    probs_similarity = tf.matmul(id_vec, tf.transpose(norm_movie_vec))
    # print(probs_similarity)
    # 取前topk的电影
    _, indices = tf.nn.top_k(probs_similarity, k=topk, sorted=True)
    with tf.Session() as sess:
        indices = sess.run([indices])
    indices = indices[0].reshape(-1).tolist()[1:]
    #     print(movies_source[movies_source["MovieID"].isin(indices)]["Title"])

    #     rec_title = movies_source[movies_source["MovieID"].isin(indices)]["Title"].values.tolist()

    print("您看的电影是：{}".format(id2title[movie_id]))
    print("以下是给您的推荐：")
    for indice in indices:
        print(indice, ":", id2title[indice])


indices = rec_similar_style(234, topk=5)

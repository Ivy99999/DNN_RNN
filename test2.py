#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ivynie
import os

from dealdata import movies_source

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)

file_path = "data"
id2title = pd.Series(movies_source["Title"].values, index=movies_source["MovieID"]).to_dict()


# 看过这个电影的人还喜欢什么电影
def view_also_view(movie_id, topk=20):
    """
    首先选出喜欢某个电影的top_k个人，得到这几个人的用户特征向量。
    然后计算这几个人对所有电影的评分,选择每个人评分最高的电影作为推荐,同样加入了随机选择
    """
    movie_vec = tf.constant(np.loadtxt(os.path.join(file_path, "data_vec", "movie_vec.txt"), dtype=np.float32))
    user_vec = tf.constant(np.loadtxt(os.path.join(file_path, "data_vec", "user_vec.txt"), dtype=np.float32))
    id_vec = tf.nn.embedding_lookup(movie_vec, np.array([[movie_id]]))
    id_vec = tf.reshape(id_vec, [-1, 256])
    probs_similarity = tf.matmul(id_vec, tf.transpose(user_vec))
    _, indices = tf.nn.top_k(probs_similarity, k=topk, sorted=True)
    indices = tf.reshape(indices, [-1, 1])
    top_user_vec = tf.nn.embedding_lookup(user_vec, indices)
    top_user_vec = tf.reshape(top_user_vec, [-1, 256])

    sim_dist = tf.matmul(top_user_vec, tf.transpose(movie_vec))

    _, top_indices = tf.nn.top_k(sim_dist, k=2)
    top_indices = tf.reshape(top_indices, [-1])
    with tf.Session() as sess:
        indices = sess.run([top_indices])
    indices = indices[0].tolist()[1:]

    print("看过的电影是：{}".format(id2title[movie_id]))
    print("看过这个电影的人还喜欢什么电影：")
    for indice in indices:
        print(indice, ":", id2title[indice])


view_also_view(1401, topk=5)

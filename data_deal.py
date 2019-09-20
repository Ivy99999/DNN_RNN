# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ivynie

import os
import re
import warnings

import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)

file_path = "data"
# User basedata
users_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
users = pd.read_table(os.path.join(file_path, "ml-1m/users.dat"), names=users_title, sep="::")
# data basedata
movies_title = ['MovieID', 'Title', 'Genres']
movies = pd.read_table(os.path.join(file_path, "ml-1m/movies.dat"), names=movies_title, sep="::")

## 用户ID、电影ID、评分和时间戳
ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
ratings = pd.read_table(os.path.join(file_path, "ml-1m/ratings.dat"), names=ratings_title, sep='::')
# 性别词典
gender_map = {"F": 0, "M": 1}
# 年龄映射
age_map = {value: id for id, value in enumerate(set(users["Age"]))}
# print(age_map)
movies_source = movies.copy()


def data_process(users, movies, ratings):
    users["Gender"] = users["Gender"].map(gender_map)
    users["Age"] = users["Age"].map(age_map)
    movies["Title"] = movies["Title"].map(lambda x: re.sub("\(\d+\)", "", x).strip())

    def seq_process(df, col, sep):
        '''
        :param df: 传入dataframe
        :param col: 传入列名
        :param sep: 传入分隔符
        :return: 返回等长的映射id序列
        '''
        col_set = set()
        col_max_len = 0
        source_sep_list = []
        for val in df[col].values:
            sep_l = val.split(sep)
            # print(sep_l)
            col_set.update(sep_l)
            source_sep_list.append(sep_l)
            if len(sep_l) > col_max_len:
                col_max_len = len(sep_l)
        # 长度不足时，增加<PAD>
        col_set.add('<PAD>')
        col2id = {value: id for id, value in enumerate(col_set)}
        dest_sep_list = [[col2id[t] for t in l] + [col2id["<PAD>"]] * (col_max_len - len(l)) for l in source_sep_list]
        df[col] = dest_sep_list
        # print(col + "列的最大长度序列为：", col_max_len)
        # print(col + "列的字典表为：", col2id)
        return df, col2id, col_max_len

    movies, title2id, title_maxlen = seq_process(movies, "Title", " ")
    movies, genre2id, genre_maxlen = seq_process(movies, "Genres", "|")
    data = pd.merge(pd.merge(ratings, users), movies)
    return data, title2id, title_maxlen, genre2id, genre_maxlen


def get_batches(Xs, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end]

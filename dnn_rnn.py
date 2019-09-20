import tensorflow as tf
import os
from data_deal import data_process
import pandas as pd
file_path = "data"
#User basedata
users_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
users = pd.read_table(os.path.join(file_path, "ml-1m/users.dat"), names=users_title, sep="::")
#data basedata
movies_title = ['MovieID', 'Title', 'Genres']
movies = pd.read_table(os.path.join(file_path, "ml-1m/movies.dat"), names=movies_title, sep="::")

## 用户ID、电影ID、评分和时间戳
ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
ratings = pd.read_table(os.path.join(file_path, "ml-1m/ratings.dat"), names=ratings_title, sep='::')

data, title2id, title_maxlen, genre2id, genre_maxlen = data_process(users, movies, ratings)
vocab_uid = max(data["UserID"].unique()) + 1
vocab_gender = max(data["Gender"].unique()) + 1
vocab_age = max(data["Age"].unique()) + 1
vocab_job = max(data["OccupationID"].unique()) + 1
vocab_mid = max(data["MovieID"].unique()) + 1
vocab_title = len(title2id)
vocab_genre = len(genre2id)
# print(vocab_title)
# print(vocab_genre)

class DNNText(object):
    def __init__(self,args):
        # 参数设置
        self.emb_dim = args.emb_dim
        self.hidden_size = args.hidden_size
        self.genre_f = args.genre_f
        self.n_neurons = args.n_neurons  # 滑动2,3,4,5个单词
        self.n_layers = args.n_layers  # 卷积核数
        self.dropout_keep_prob = args.dropout_keep_prob
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.display_steps = args.display_steps
        self.dnn()
        # self.itemdnn()


    def dnn(self):
        # 定义placeholder
        with tf.name_scope("input_placeholder"):
            self.uid = tf.placeholder(tf.int32, shape=[None, 1], name="uid")
            self.gender = tf.placeholder(tf.int32, shape=[None, 1], name="user_gender")
            self.age = tf.placeholder(tf.int32, shape=[None, 1], name="user_age")
            self.job = tf.placeholder(tf.int32, shape=[None, 1], name="user_job")
            self.mid = tf.placeholder(tf.int32, shape=[None, 1], name="mid")
            self.title = tf.placeholder(tf.int32, shape=[None, 15], name="movie_title")
            self.genre = tf.placeholder(tf.int32, shape=[None, 6], name="movie_genre")
            self.target = tf.placeholder(tf.float32, shape=[None, 1], name="ratings")


        with tf.name_scope("u_embedding"):
            uid_embedding = tf.Variable(tf.random_normal([vocab_uid, self.emb_dim], 0, 1), name="user_embedding")
            uid_embed = tf.nn.embedding_lookup(uid_embedding, self.uid, name="user_embed")
            gender_embedding = tf.Variable(tf.random_normal([vocab_gender, self.emb_dim // 2], 0, 1),
                                           name="gender_embedding")
            gender_embed = tf.nn.embedding_lookup(gender_embedding, self.gender, name="gender_embed")
            age_embedding = tf.Variable(tf.random_normal([vocab_age, self.emb_dim // 2], 0, 1), name="age_embedding")
            age_embed = tf.nn.embedding_lookup(age_embedding, self.age, name="age_embed")
            job_embedding = tf.Variable(tf.random_normal([vocab_job, self.emb_dim // 2], 0, 1), name="job_embedding")
            job_embed = tf.nn.embedding_lookup(job_embedding, self.job, name="job_embed")

        with tf.name_scope("user_nn"):
            # uid_embed: (batch_size, seq_len, emb_dim)
            uid_fc = tf.layers.dense(uid_embed, self.emb_dim, activation=tf.nn.relu)
            gender_fc = tf.layers.dense(gender_embed, self.emb_dim, activation=tf.nn.relu)
            age_fc = tf.layers.dense(age_embed, self.emb_dim, activation=tf.nn.relu)
            job_fc = tf.layers.dense(job_embed, self.emb_dim, activation=tf.nn.relu)

            # 对上述数据进行拼接
            u_cat = tf.concat([uid_fc, gender_fc, age_fc, job_fc], axis=-1)
            #         u_cat = tf.nn.tanh(tf.layers.dense(u_cat, hidden_size))
            u_cat = tf.layers.dense(u_cat, self.hidden_size, activation=tf.nn.tanh)
            self.uinfo = tf.reshape(u_cat, [-1, self.hidden_size])
        with tf.name_scope("m_embedding"):
            mid_embedding = tf.Variable(tf.random_normal([vocab_mid, self.emb_dim], 0, 1))
            mid_embed = tf.nn.embedding_lookup(mid_embedding, self.mid)
            #修改的初始化向量问题
            genre_embedding = tf.Variable(tf.random_normal([vocab_genre, self.emb_dim // 2], 0, 1))
            genre_embed = tf.nn.embedding_lookup(genre_embedding, self.genre)
            if self.genre_f == "sum":
                genre_embed = tf.reduce_sum(genre_embed, axis=1, keepdims=True)
            elif self.genre_f == "mean":
                genre_embed = tf.reduce_mean(genre_embed, axis=1, keepdims=True)

            # 关于标题的cnn
            with tf.name_scope("title_cnn"):
                title_embedding = tf.Variable(tf.random_normal([vocab_title, self.emb_dim], 0, 1))
                title_embed = tf.nn.embedding_lookup(title_embedding, self.title)
                # print(title_embed.shape())


                def lstm_cell():
                    return tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, state_is_tuple=True)

                def dropout_cell():
                    cell = lstm_cell()
                    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
                # for layer in range(self.n_layers):
                #     print(layer)

                cells = [dropout_cell() for layer in range(self.n_layers)]
                print(cells)
                rnn_cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                outputs, states = tf.nn.dynamic_rnn(rnn_cells, inputs=title_embed, dtype=tf.float32)
                print(outputs)
                # h_dropout=outputs
                h_dropout = outputs[:, -1, :]  # 取最后一个时序输出结果
                h_dropout = tf.expand_dims(h_dropout, 1)
                # print(h_dropout.shape())


        with tf.name_scope("m_nn"):
            mid_fc = tf.layers.dense(mid_embed, self.emb_dim, activation=tf.nn.relu)
            genre_fc = tf.layers.dense(genre_embed, self.emb_dim, activation=tf.nn.relu)
            print(mid_fc)
            print(gender_fc)
            print(h_dropout)
            m_cat = tf.concat([mid_fc, genre_fc, h_dropout], axis=-1)
            m_fc = tf.layers.dense(m_cat, self.hidden_size, activation=tf.nn.tanh)
            self.m_fc = tf.reshape(m_fc, [-1, self.hidden_size])
        with tf.name_scope("loss"):

            self.inference = tf.reduce_sum(tf.multiply(self.uinfo, self.m_fc), axis=-1, keepdims=True, name="inference")
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.target, self.inference))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.loss)
#         #





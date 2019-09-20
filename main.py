import argparse
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from data_deal import data_process, get_batches
import tensorflow as tf
import numpy as np
from dnn_rnn import DNNText

parser = argparse.ArgumentParser(description='DNN for Classify')
parser.add_argument('--emb_dim', type=int, default=128, help='Percentage of the training data to use for validation')
parser.add_argument('--hidden_size', type=int, default=256, help='train data source')
parser.add_argument('--genre_f', type=str, default='sum', help='test data source')
parser.add_argument('--n_neurons', type=int, default=128, help='label')
parser.add_argument('--n_layers', type=int, default=2, help='length')
parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help='#sample of each minibatch')
parser.add_argument('--lr', type=float, default= 0.0001, help='#')
parser.add_argument('--num_epochs', type=int, default=5, help='#dim of hidden state')
parser.add_argument('--batch_size', type=float, default=256, help='dropout keep_prob')
parser.add_argument('--display_steps', type=int, default=600, help='dropout keep_prob')
args = parser.parse_args()

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
# 划分训练集和测试集， 同时生成batch data
train, test = train_test_split(data, test_size=0.2, random_state=123)

batch_data = next(get_batches(train, 3))

# # 训练网络
model_path = os.path.join(file_path, "model/model.ckpt")  # 模型权重的保存地址

dnn = DNNText(args)





#
# with tf.Graph().as_default():
#     sess = tf.Session()
#     with sess.as_default():
#         # dnn = DNNText(args)
#         print(dnn)
#         loss_summary = tf.summary.scalar("loss", dnn.loss)
#         timestamp = str(int(time.time()))
#         out_dir = os.path.join(file_path, "runs", timestamp)
#         train_summary_dir = os.path.join(out_dir, "summaries", "train")
#         train_summary_writer = tf.summary.FileWriter(train_summary_dir, graph=sess.graph)
#
#         # test op to write logs to Tensorboard
#         test_summary_dir = os.path.join(out_dir, "summaries", "test")
#         test_summary_writer = tf.summary.FileWriter(test_summary_dir, graph=sess.graph)
#         sess.run(tf.global_variables_initializer())
#         # Saver op to save and restore all the variables
#         saver = tf.train.Saver()
#
#         for epoch in range(args.num_epochs):
#             train, test = train_test_split(data, test_size=0.2)
#             train_batch_iterator = get_batches(train, args.batch_size)
#             test_batch_iterator = get_batches(test, args.batch_size)
#             for step in range(len(train) // args.batch_size):
#                 batch_data = next(train_batch_iterator)
#                 batchX, batchY = batch_data.drop("Rating", axis=1), batch_data["Rating"]
#                 feed_dict = {
#                     dnn.uid: np.reshape(batchX["UserID"].values, [-1, 1]),
#                     dnn.gender: np.reshape(batchX["Gender"].values, [-1, 1]),
#                     dnn.age: np.reshape(batchX["Age"].values, [-1, 1]),
#                     dnn.job: np.reshape(batchX["OccupationID"].values, [-1, 1]),
#                     dnn.mid: np.reshape(batchX["MovieID"].values, [-1, 1]),
#                     dnn.title: np.array(batchX["Title"].values.tolist()),
#                     dnn.genre: np.array(batchX["Genres"].values.tolist()),
#                     dnn.target: np.reshape(batchY.values, [-1, 1]),
#                 }
#                 summaries, loss, _, rating_score, preidct = sess.run(
#                     [loss_summary, dnn.loss, dnn.train_op, dnn.target, dnn.inference],
#                     feed_dict=feed_dict)
#                 # print("loss",loss)
#                 if step % args.display_steps == 0:
#
#                     print(" Epoch {:>3} Batch {:>4}/{}  train_loss={:.3f}".format( epoch, step,
#                                                                                      len(train) // args.batch_size, loss))
#             # 进行测试
#             for step in range(len(test) // args.batch_size):
#                 batch_data = next(test_batch_iterator)
#                 batchX, batchY = batch_data.drop("Rating", axis=1), batch_data["Rating"]
#                 feed_dict = {
#                     dnn.uid: np.reshape(batchX["UserID"].values, [-1, 1]),
#                     dnn.gender: np.reshape(batchX["Gender"].values, [-1, 1]),
#                     dnn.age: np.reshape(batchX["Age"].values, [-1, 1]),
#                     dnn.job: np.reshape(batchX["OccupationID"].values, [-1, 1]),
#                     dnn.mid: np.reshape(batchX["MovieID"].values, [-1, 1]),
#                     dnn.title: np.array(batchX["Title"].values.tolist()),
#                     dnn.genre: np.array(batchX["Genres"].values.tolist()),
#                     dnn.target: np.reshape(batchY.values, [-1, 1]),
#                 }
#                 summaries, loss, _ = sess.run([loss_summary, dnn.loss, dnn.train_op], feed_dict=feed_dict)
#                 test_summary_writer.add_summary(summaries, step)
#
#                 if step % args.display_steps == 0:
#                     # now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                     print("Epoch {:>3} Batch {:>4}/{}  test_loss={:.3f}".format(epoch, step,
#                                                                                 len(test) // args.batch_size, loss))
#     # Save model weights to disk
#     saver.save(sess, save_path=model_path)
#     print("模型已经训练完成，同时已经保存到磁盘！")


def save_weights():
    """保存用户向量和电影向量结果"""
    with tf.Session() as sess:
        # 构造网络图
        saver = tf.train.import_meta_graph(os.path.join(file_path, "model/model.ckpt.meta"))
        # 加载参数
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(file_path, "model")))

        # 访问图
        graph = tf.get_default_graph()
        uid = graph.get_tensor_by_name('input_placeholder/uid:0')
        print(uid)
        gender = graph.get_tensor_by_name('input_placeholder/user_gender:0')
        age = graph.get_tensor_by_name('input_placeholder/user_age:0')
        job = graph.get_tensor_by_name('input_placeholder/user_job:0')
        mid = graph.get_tensor_by_name('input_placeholder/mid:0')
        title = graph.get_tensor_by_name('input_placeholder/movie_title:0')
        genre = graph.get_tensor_by_name('input_placeholder/movie_genre:0')
        # inference = graph.get_tensor_by_name('inference:0')
        uinfo = graph.get_tensor_by_name('user_nn/Reshape:0')
        m_fc = graph.get_tensor_by_name('m_nn/Reshape:0')
#         # print(uinfo)
#         # print(m_fc)
#
        feed_dict = {
            uid: np.reshape(users["UserID"].values, [-1, 1]),
            gender: np.reshape(users["Gender"].values, [-1, 1]),
            age: np.reshape(users["Age"].values, [-1, 1]),
            job: np.reshape(users["OccupationID"].values, [-1, 1]),
            mid: np.reshape(movies["MovieID"].values, [-1, 1]),
            title: np.array(movies["Title"].values.tolist()),
            genre: np.array(movies["Genres"].values.tolist()),
        }
        user_vec, movie_vec = sess.run([uinfo,m_fc], feed_dict=feed_dict)
        np.savetxt(os.path.join(file_path, "data_vec_rnn", "user_vec.txt"), user_vec, fmt="%0.4f")
        np.savetxt(os.path.join(file_path, "data_vec_rnn", "movie_vec.txt"), movie_vec, fmt="%0.4f")
print(save_weights())
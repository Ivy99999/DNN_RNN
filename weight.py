#
#
# def save_weights():
#     """保存用户向量和电影向量结果"""
#     with tf.Session() as sess:
#         # 构造网络图
#         saver = tf.train.import_meta_graph(os.path.join(file_path, "model/model.ckpt.meta"))
#         # 加载参数
#         saver.restore(sess, tf.train.latest_checkpoint(os.path.join(file_path, "model")))
#
#         # 访问图
#         graph = tf.get_default_graph()
#         #         for op in graph.get_operations():
#         #             print(str(op.name))
#
#         uid = graph.get_tensor_by_name('input_placeholder/uid:0')
#         gender = graph.get_tensor_by_name('input_placeholder/user_gender:0')
#         age = graph.get_tensor_by_name('input_placeholder/user_age:0')
#         job = graph.get_tensor_by_name('input_placeholder/user_job:0')
#         mid = graph.get_tensor_by_name('input_placeholder/mid:0')
#         title = graph.get_tensor_by_name('input_placeholder/movie_title:0')
#         genre = graph.get_tensor_by_name('input_placeholder/movie_genre:0')
#         inference = graph.get_tensor_by_name('inference:0')
#         uinfo = graph.get_tensor_by_name('user_nn/Reshape:0')
#         m_fc = graph.get_tensor_by_name('m_nn/Reshape:0')
#         # print(uinfo)
#         # print(m_fc)
#
#         feed_dict = {
#             uid: np.reshape(users["UserID"].values, [-1, 1]),
#             gender: np.reshape(users["Gender"].values, [-1, 1]),
#             age: np.reshape(users["Age"].values, [-1, 1]),
#             job: np.reshape(users["OccupationID"].values, [-1, 1]),
#             mid: np.reshape(movies["MovieID"].values, [-1, 1]),
#             title: np.array(movies["Title"].values.tolist()),
#             genre: np.array(movies["Genres"].values.tolist()),
#         }
#         user_vec, movie_vec = sess.run([uinfo, m_fc], feed_dict=feed_dict)
#         np.savetxt(os.path.join(file_path, "data_vec", "user_vec.txt"), user_vec, fmt="%0.4f")
#         np.savetxt(os.path.join(file_path, "data_vec", "movie_vec.txt"), movie_vec, fmt="%0.4f")
# save_weights()
# # 构建电影id和电影名称的字典表
# id2title = pd.Series(movies_source["Title"].values, index=movies_source["MovieID"]).to_dict()
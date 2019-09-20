# import json
#
# import requests
#
# from elasticsearch import RequestsHttpConnection, Elasticsearch
#
#
# class MyConnection(RequestsHttpConnection):
#     def __init__(self, *args, **kwargs):
#         proxies = kwargs.pop('proxies', {})
#         super(MyConnection, self).__init__(*args, **kwargs)
#         self.session.proxies = proxies
#
#
# esclient_dns = Elasticsearch(["http://172.16.38.166:9200"],
#                              connection_class=MyConnection,
#                              proxies={'http': None},
#                              timeout=600,
#                              max_retries=10,
#                              retry_on_timeout=True)
# print(esclient_dns)
#
# CREATE_BODY = {
#     "settings": {
#         "number_of_shards": 4,  # 数据自动会分成四片存放在不同的节点，提高数据检索速度
#         "number_of_replicas": 0
#     # 创建0个副本集，如果ES是集群，设置多副本会自动将副本创建到多个节点；设置多副本可以增加数据库的安全性，但是插数据的时候，会先向主节点插入数据，之后再向其余副本同步，会降低插入数据速度，差不多会降低1.4到1.6倍
#     },
#
#     "mappings": {
#         "*": {  # 文档类型为dns_info
#             "properties": {
#                 "corpus_info_id": {
#                     "type": "keyword"
#                 },
#                 "text": {
#                     "type": "keyword"
#                 },
#                 "task_id": {
#                     "type": "keyword"
#                 },
#                 "sub_task_id": {
#                     "type": "keyword"
#                 },
#                 "create_time": {
#                     "type": "date",
#                     "format": "yyyy-MM-dd HH:mm:ss"
#                 },
#                 "annotation_time": {
#                     "type": "date",
#                     "format": "yyyy-MM-dd HH:mm:ss"
#                 },
#                 "sentence_status": {
#                     "type": "integer"
#                 },
#                 "executor": {
#                     "type": "keyword"
#                 },
#                 "offset": {
#                     "type": "integer"
#                 },
#                 "index": {
#                     "type": "integer"
#                 },
#                 "relation": {
#                     "type": "keyword"
#                 },
#                 "entityRelation": {
#                     "type": "keyword"
#                 },
#                 "back_reason": {
#                     "type": "keyword"
#                 }
#             }
#         },
#         "_default_": {
#             "dynamic": "strict"  # 限定不能多增加或减少字段，如果多增加或减少字段就会抛出异常；dynamic是处理遇到未知字段情况的，设置为“strict”是说当遇到未知字段时抛出异常
#         }
#     }
# }
#
# header = {'Content-Type': 'application/json'}
# DNS_API = "http://172.16.38.166:9200/corpus_annotation"  # dns_test_v2为索引名
# try:
#     resp = requests.put(DNS_API, headers=header, data=json.dumps(CREATE_BODY))
#     if resp.status_code == 200:
#         print(u"成功建立索引")
# except Exception as e:
#     print(e, u"建立索引失败")

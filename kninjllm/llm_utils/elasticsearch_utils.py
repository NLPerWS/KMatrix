from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

class ElasticsearchTool:
    def __init__(self, host, port,scheme):
        self.es = Elasticsearch([{'host': host, 'port': port,'scheme':scheme}])

    def create_index(self, index_name, mappings):
        self.es.indices.create(index=index_name, body=mappings)

    def delete_index(self, index_name):
        self.es.indices.delete(index=index_name)

    def index_document(self, index_name, doc_type, document, doc_id=None):
        self.es.index(index=index_name, body=document, id=doc_id)

    def get_document(self, index_name, doc_id):
        return self.es.get(index=index_name, id=doc_id)

    def update_document(self, index_name, doc_id, updated_document):
        self.es.update(index=index_name, id=doc_id, body={"doc": updated_document})

    def delete_document(self, index_name, doc_id):
        self.es.delete(index=index_name, id=doc_id)

    def bulk_index_documents(self, index_name, documents):
        actions = [
            {
                "_index": index_name,
                "_id": doc['id'],
                "_source": doc
            }
            for doc in documents
        ]
        bulk(self.es, actions)
        
    def query_documents_by_size(self, index_name, size=10):
        body = {
            "size": size,
            "query": {
                "match_all": {}
            }
        }
        return self.es.search(index=index_name, body=body)
        
if __name__ == "__main__":
    es_tool = ElasticsearchTool("127.0.0.1", 9200,'http')

    retrieved_doc = es_tool.query_documents_by_size('default', 10)
    print(retrieved_doc)

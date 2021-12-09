import os
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from absl import app, flags, logging
from absl.flags import FLAGS

def main(_argv):
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
    document_store.delete_documents()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

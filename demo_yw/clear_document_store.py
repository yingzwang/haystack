import os
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from absl import app, flags, logging
from absl.flags import FLAGS

def main(_argv):
    # Not needed. In init_document_store.py, we always delete existing documents first.
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
    document_store.delete_documents()
    print(f"In document store: {document_store.get_document_count()} documents.")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

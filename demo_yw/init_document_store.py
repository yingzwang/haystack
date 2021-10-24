import os
from typing import Callable, Dict, List, Optional, Tuple, Union, Generator
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.preprocessor.utils import convert_files_to_dicts
from haystack.preprocessor import PreProcessor
from haystack.utils import launch_es
from absl import app, flags, logging
from absl.flags import FLAGS
flags.DEFINE_string('doc_dir', './data/ziggo_wifi_en', 'Directory of documents (txt files)')
# FLAGS([''])


def load_and_clean_documents(doc_dir: str):
    # TODO: add function "add_period_to_end_of_sentence", add a period to the end of a bullet point sentence,
    #  e.g., for '1_smartwifi_pods.txt', add period to the end of sentences.
    raw_docs = convert_files_to_dicts(dir_path=doc_dir) # just read all text files, do not clean anything yet
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=100,
        split_respect_sentence_boundary=True,
    )
    docs = preprocessor.process(raw_docs)
    logging.info(f"Before PrepProcessor: {len(raw_docs)} documents.")
    logging.info(f"After PrepProcessor: {len(docs)} documents.")
    return docs


def main(_argv):
    # launch_es() # Do not start elasticsearch in the python script.
    FLAGS.doc_dir = os.path.expanduser(FLAGS.doc_dir)
    docs = load_and_clean_documents(doc_dir=FLAGS.doc_dir)
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
    document_store.write_documents(docs)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

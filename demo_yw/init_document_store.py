import os
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.utils.preprocessing import convert_files_to_dicts
from haystack.nodes.preprocessor import PreProcessor
from absl import app, flags, logging
from absl.flags import FLAGS
flags.DEFINE_string('doc_dir', './data/ziggo_wifi_en', 'Directory of documents (txt, pdf, docx)')
# FLAGS([''])


def load_and_clean_documents(doc_dir: str):
    # TODO: add function "add_period_to_end_of_sentence", add a period to the end of a bullet point sentence,
    #  e.g., for '1_smartwifi_pods.txt', add period to the end of sentences.
    # Note: split_paragraphs=False (default) is good. Do not split paragraphs, better keep it a long document.
    raw_docs = convert_files_to_dicts(dir_path=doc_dir) # just read all text files, do not clean anything yet
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=1000, # Note: keep it the same as the pipelines.yaml in rest_api
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
    document_store.delete_documents()
    document_store.write_documents(docs)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

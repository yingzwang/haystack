## Quick Demo Local (yw notes)

Run Haystack quick demo in local conda environment.

#### Create conda environment
```bash
conda create -n haystack-dev python=3.7.4    
conda activate haystack-dev
```

#### Install dependencies
```bash
git clone https://github.com/deepset-ai/haystack.git
cd haystack
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
python3 -c "import nltk;nltk.download('punkt')"
cd ui; pip install -r requirements.txt
```

#### Run Elasticsearch with GOT articles (optional)
```bash
docker run -dp 9200:9200 -e "discovery.type=single-node" deepset/elasticsearch-game-of-thrones
```

---
#### Run Elasticsearch with an empty instance
```bash
docker run -dp 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2
```

#### Load documents to elasticsearch (demo 1: WiFi trouble shooting)
```bash
python ./demo_yw/init_document_store.py --doc_dir="./data/ziggo_wifi_en"
```

#### Run Haystack API
```bash
gunicorn rest_api.application:app -b 0.0.0.0 -k uvicorn.workers.UvicornWorker --workers 1 --timeout 180
```

#### Start UI
```bash
streamlit run ./ui/webapp_ziggo.py
```

---
#### Load documents to elasticsearch (demo 2: financial reports)
```bash
python ./demo_yw/init_document_store.py --doc_dir="./data/ziggo_pdf"
```

**Note**: We use following ports
* Haystack API: listens on port 8000
* DocumentStore (Elasticsearch): listens on port 9200
* Streamlit UI: listens on port 8501


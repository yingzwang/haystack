## Quick Demo Local (yw notes)

Run Haystack quick demo in local conda environment.

**1. Create conda environment**
```bash
conda create -n haystack-dev python=3.7.4    
conda activate haystack-dev
```

**2. Install dependencies**
```bash
git clone https://github.com/deepset-ai/haystack.git
cd haystack
pip install -r requirements.txt
pip install -e .
# install streamlit
cd ui
pip install -r requirements.txt
```

**3. Run Elasticsearch container (with Game of Thrones articles)**
```bash
docker run -dp 9200:9200 -e "discovery.type=single-node" deepset/elasticsearch-game-of-thrones
```

**4. Run Haystack API**
```bash
gunicorn rest_api.application:app -b 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker -t 300
```

**5. Start UI**
```bash
streamlit run webapp.py
```

**Note**: We use following ports
* Haystack API: listens on port 8000
* DocumentStore (Elasticsearch): listens on port 9200
* Streamlit UI: listens on port 8501


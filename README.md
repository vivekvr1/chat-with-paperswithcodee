# RAG-based Assistant to Chat with Papers With Code

This repository contains the code for building a RAG-based assistant to chat with Papers With Code using Streamlit.

## Requirements

- A GCP account with VertexAI and Cloud Run services activated
- An OpenAI API key
- A free account on [Upstash](https://upstash.com/) (serverless database)
- Pulumi installed and configured

## 1. Indexing

To index data into the vector DB, you first need to create an index on Upstash and fill in the credentials in the `.env` file:

```plaintext
UPSTASH_VECTOR_REST_URL=...
UPSTASH_VECTOR_REST_TOKEN=...
OPENAI_API_KEY=...
```

# APP LEVEL COMMANDS

```
python src/paperswithcode.py
python src/index_papers.py test --query "openai" --max_papers 5
python src/index_papers.py test-upstash
python src/index_papers.py index --query "openai" --max_papers 20
python src/brag.py
streamlit run bapp.py --theme.primaryColor "#135aaf"

docker build -t streamlit-papers-app .
docker run -d  --name papers-chat   -p 8501:8501  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d '=' -f2)" -e UPSTASH_VECTOR_REST_URL="$(grep UPSTASH_VECTOR_REST_URL .env | cut -d '=' -f2)"   -e UPSTASH_VECTOR_REST_TOKEN="$(grep UPSTASH_VECTOR_REST_TOKEN .env | cut -d '=' -f2)"  streamlit-papers-app
OR
docker run -d --name papers-chat -p 8501:8501 --env-file .env streamlit-papers-app
OR
docker run -d   --name papers-chat   -p 8501:8501   streamlit-papers-app

docker exec -it container_id bash
```

# Google Cloud Run

```
gcloud auth configure-docker
gcloud builds submit --tag gcr.io/krishai-455907/rag --timeout=2h
```

# Google App Engine

```
gcloud auth configure-docker
gcloud builds submit --tag gcr.io/krishai-455907/rag --timeout=2h
gcloud config get-value project\n
gcloud projects add-iam-policy-binding krishai-455907     --member="user:paul.visionai@gmail.com"     --role="roles/cloudbuild.builds.editor"
gcloud services enable cloudbuild.googleapis.com\n
gcloud auth list
gcloud config list
gcloud projects get-iam-policy $(gcloud config get-value project)
gcloud builds submit --tag gcr.io/krishai-455907/rag --timeout=2h
gcloud services enable appengine.googleapis.com
gcloud app deploy
```

# PULUMI
```
pip install pulumi pulumi-gcp pulumi-docker
pulumi stack init dev
pulumi up
```
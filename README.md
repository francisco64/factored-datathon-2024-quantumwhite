# Factored Datathon 2024 - QuantumWhite

## Overview

This repository contains a prototype data and machine learning workflow built for the Factored Datathon 2024. The project explores the GDELT dataset to ingest global event and knowledge graph data into Google BigQuery, analyze COVID-era events, cluster related actors and events, and retrieve representative news articles for summarization.

The codebase is organized as a collection of pipeline scripts, SQL queries, notebooks, and cloud deployment artifacts. It is best understood as a datathon / research prototype rather than a complete production application.

## Key Features

- Apache Beam pipeline for downloading GDELT `.zip` files, extracting CSV content, uploading data to Google Cloud Storage, and loading it into BigQuery.
- Support for both GDELT Events and GKG schemas through separate schema modules.
- Dataflow Flex Template metadata and Dockerfile for packaging the ingestion pipeline.
- Exploratory BigQuery SQL for COVID-era event analysis, time-series analysis, correlation analysis, and actor-event relationships.
- KMeans-based exploratory clustering over actor-event features using BigQuery ML.
- Relevant news retrieval workflow that combines token-based features, Universal Sentence Encoder embeddings, KMeans clustering, t-SNE projection, LDA topic modeling, and Gemini summaries.
- Cloud Function stub that triggers a Cloud Run-style processing service when new GCS objects arrive.
- Presentation and architecture diagrams documenting the intended end-to-end system.

## Technical Approach

The project has four main parts:

1. **GDELT ingestion pipeline**
   - `data_pipeline/gdelt2bq.py` fetches file links from a GDELT index page, filters URLs by include/exclude terms, optionally restricts processing to the previous day, downloads zip files, extracts CSV files, writes them to GCS, and loads them into BigQuery.
   - The pipeline uses Apache Beam with the Dataflow runner and Google Cloud clients for Storage and BigQuery.
   - BigQuery schemas are defined in `data_pipeline/gdelt_schemas/`.

2. **Exploratory analytics**
   - SQL files in `data_analysis_and_exploratoryML/sql_queries/` analyze COVID-related events, disturbances in the United States, actor-event relationships, time-series behavior, and correlations.
   - `covid_major_events.csv` provides supporting event context for analysis.
   - `graphConnectionactor-event.py` derives actor-event graph rows from a local cluster export.

3. **KMeans exploratory analysis**
   - `kmeans_for_exploratory_analysis/bq_kmeans.sql` defines a BigQuery ML KMeans model using cosine distance.
   - `kmeans_for_exploratory_analysis/kmeans_training_dataset.ipynb` supports the training dataset workflow.

4. **Relevant news retrieval system**
   - `relevant_news_retreival_system/main.py` exposes a Flask endpoint intended to receive GCS object notifications.
   - For matching GKG CSV files, it reads the data, filters rows by `NUMARTS`, extracts text from source URLs, generates Universal Sentence Encoder embeddings, creates multi-hot token vectors from GKG fields, clusters normalized vectors with KMeans, writes t-SNE coordinates to BigQuery, selects representative articles, applies LDA topic modeling, and summarizes selected text with Gemini through Vertex AI.
   - `relevant_news_retreival_system/cloud_functions/main.py` contains a Cloud Function entry point for forwarding GCS events to the processing service.

## Tech Stack

- **Language:** Python, SQL
- **Data processing:** Apache Beam, Google Cloud Dataflow
- **Cloud services:** Google Cloud Storage, BigQuery, BigQuery ML, Cloud Functions, Cloud Run-style Flask service, Vertex AI
- **Machine learning / NLP:** scikit-learn, TensorFlow Hub Universal Sentence Encoder, LDA topic modeling, KMeans clustering, t-SNE
- **Data tooling:** pandas, NumPy, pandas-gbq, BeautifulSoup, requests
- **Packaging / deployment:** Docker, Dataflow Flex Templates
- **Artifacts:** Jupyter notebooks, PDF/PPSX presentation, architecture images

## Techniques Used

- Batch data ingestion from public GDELT indexes
- Zip extraction and tab-delimited CSV loading
- BigQuery table creation and batch load jobs
- BigQuery ML KMeans clustering with cosine distance
- Multi-hot encoding of GDELT token fields
- Text extraction from article URLs
- Sentence embeddings with Universal Sentence Encoder
- Vector normalization and KMeans clustering
- t-SNE dimensionality reduction for visualization data
- LDA topic modeling over selected article text
- LLM-based summarization through Vertex AI Gemini
- Cloud-triggered processing using GCS events

## Repository Structure

```text
.
|-- README.md
|-- data_pipeline/
|   |-- gdelt2bq.py
|   |-- Dockerfile
|   |-- template_spec.json
|   |-- requirements.txt
|   `-- gdelt_schemas/
|-- data_analysis_and_exploratoryML/
|   |-- covid_major_events.csv
|   |-- graphConnectionactor-event.py
|   `-- sql_queries/
|-- kmeans_for_exploratory_analysis/
|   |-- bq_kmeans.sql
|   |-- kmeans_training_dataset.ipynb
|   `-- readme.md
|-- relevant_news_retreival_system/
|   |-- main.py
|   |-- requirements.txt
|   |-- krn-nlp.ipynb
|   `-- cloud_functions/
|-- images/
|-- factored-competition-2024.pdf
`-- factored-competition-2024.ppsx
```

## Getting Started

### Prerequisites

- Python 3.8+ recommended
- Google Cloud project with BigQuery, Cloud Storage, Dataflow, and Vertex AI enabled
- Google Cloud SDK installed and authenticated
- Docker, if building the Dataflow Flex Template image
- Access to the GDELT source indexes used by the pipeline

### Install Pipeline Dependencies

From the data pipeline folder:

```bash
cd data_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The relevant news retrieval requirements file is incomplete relative to the imports in `relevant_news_retreival_system/main.py`. If running that service locally, install the listed requirements and any missing packages required by the imports, such as Flask, scikit-learn, TensorFlow Hub, BeautifulSoup, pandas-gbq, and Vertex AI client libraries.

### Run the GDELT Ingestion Pipeline

The ingestion script expects runtime parameters for the target table, GCP region, project, source index URL, URL filters, schema, and daily-processing flag.

Example for GKG data:

```bash
cd data_pipeline
python gdelt2bq.py \
  --table_name "gkg-data" \
  --region "your-gcp-region" \
  --project_id "your-gcp-project-id" \
  --url_of_index "http://data.gdeltproject.org/gkg/index.html" \
  --in_url ".zip" \
  --not_in_url "counts" \
  --schema "gkg" \
  --day_before_condition 1
```

Example for Events data:

```bash
cd data_pipeline
python gdelt2bq.py \
  --table_name "events" \
  --region "your-gcp-region" \
  --project_id "your-gcp-project-id" \
  --url_of_index "http://data.gdeltproject.org/events/index.html" \
  --in_url ".export.CSV.zip" \
  --schema "events" \
  --day_before_condition 1
```

The script is configured to use the Dataflow runner, so it requires valid Google Cloud authentication and cloud resources.

### Build the Dataflow Flex Template Image

```bash
cd data_pipeline
docker build -t gcr.io/your-gcp-project-id/gdelt2bq:latest .
docker push gcr.io/your-gcp-project-id/gdelt2bq:latest
```

The included `template_spec.json` defines the runtime parameters needed by the Flex Template.

## Configuration

No `.env` file is present in the repository. The Google Cloud client libraries generally rely on Application Default Credentials.

Common configuration values needed to run or adapt the project:

```bash
GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
GCP_PROJECT_ID="your-gcp-project-id"
GCP_REGION="your-gcp-region"
GCS_BUCKET="your-gcs-bucket"
BIGQUERY_DATASET="your_bigquery_dataset"
```

Do not commit service account files, raw credentials, API keys, or local `.env` values.

## Usage

- Use `data_pipeline/gdelt2bq.py` to ingest GDELT Events or GKG data into BigQuery.
- Use the SQL files under `data_analysis_and_exploratoryML/sql_queries/` to reproduce exploratory analyses in BigQuery.
- Use `kmeans_for_exploratory_analysis/bq_kmeans.sql` as the BigQuery ML model definition for KMeans clustering.
- Use `relevant_news_retreival_system/main.py` as the Flask processing service for representative news retrieval and summarization.
- Review `factored-competition-2024.pdf`, `factored-competition-2024.ppsx`, and the `images/` folder for architecture and presentation context.

## Results / Outputs

The repository includes documentation and scripts that point to the following expected outputs:

- GDELT Events and GKG tables loaded into BigQuery.
- Processed CSV files staged in Google Cloud Storage.
- BigQuery ML KMeans model and clustered actor-event analysis.
- t-SNE projection output written to a BigQuery table named like `dimension_reduction`.
- Relevant article summaries and topic outputs written to a BigQuery table named like `relevant_news`.
- Architecture diagrams for the ingestion pipeline and relevant news retrieval system.

## Limitations

- This is a datathon prototype with project-specific cloud assumptions in the scripts.
- Some source files contain hardcoded cloud project, table, local path, or service values that should be parameterized before reuse.
- The relevant news retrieval requirements file does not list every package imported by the service.
- There is no automated test suite in the repository.
- The Cloud Function currently forwards events to a fixed service endpoint in source code; this should be moved to environment-based configuration before deployment.
- Notebook outputs and cloud resources are not enough on their own to fully reproduce the original environment.
- The folder name `relevant_news_retreival_system` contains a spelling error, but it is kept as-is to match the repository.

## Future Improvements

- Move project IDs, service URLs, dataset names, and table names into command-line arguments or environment variables.
- Add a consolidated requirements file or separate complete dependency files for each runnable service.
- Add setup instructions for creating the required BigQuery datasets, GCS buckets, and Dataflow template resources.
- Add basic tests for URL filtering, schema selection, GCS upload behavior, and BigQuery load configuration.
- Replace local absolute paths in exploratory scripts with configurable input paths.
- Add deployment notes for Cloud Run and Cloud Functions, including IAM permissions and service account requirements.

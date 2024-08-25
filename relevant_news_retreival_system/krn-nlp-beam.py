import os
import pandas as pd
import re
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
import requests
from bs4 import BeautifulSoup
import tensorflow_hub as hub
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from pandas_gbq import to_gbq
import base64
import vertexai
from vertexai.generative_models import GenerativeModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions, SetupOptions
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading Universal Sentence Encoder model...")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
logger.info("Model loaded.")

def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except requests.RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

def get_embedding(text):
    embedding = embed([text]).numpy()
    return np.array(embedding)[0]

def tokenize_and_count(df, columns):
    all_tokens = []
    for col in columns:
        tokens = df[col].dropna().str.split('[#;]').explode().str.strip()
        tokens = tokens[tokens.str.isalpha()]
        all_tokens.extend(tokens)
    return Counter(all_tokens)

def create_multihot_vector(row, selected_tokens, columns):
    tokens = set()
    for col in columns:
        if pd.notna(row[col]):
            col_tokens = set(re.split('[#;]', row[col]))
            tokens.update(col_tokens.intersection(selected_tokens))
    multihot_vector = [1 if token in tokens else 0 for token in sorted_selected_tokens]
    return multihot_vector





        def gemini(txt):
            vertexai.init(project="factoreddatathon2014", location="us-central1")
            model = GenerativeModel("gemini-1.5-flash-001")
            responses = model.generate_content(
                [f'I am trying to get the most important information for a news summary for the next text, be extremly consize and accurate with your words, not more than 3 phrases. If the following text is not worth of summarizing put this exact words #nosummary#, do not answer anything:{txt}'],
                generation_config={"max_output_tokens": 100, "temperature": 1, "top_p": 0.95,},
                stream=True,
            )
            response_text = ""
            for response in responses:
                response_text += response.text
            return response_text

        closest_to_cluster['extracted_text'] = closest_to_cluster['SOURCEURLS'].apply(
            lambda urls: get_text_from_url(urls.split('<UDIV>')[0].strip())
        )

        closest_to_cluster['summary'] = closest_to_cluster['extracted_text'].apply(gemini)

        relevant_news_table_destination = f'{dataset_id}.{relevant_news_table_id}'
        to_gbq(closest_to_cluster, destination_table=relevant_news_table_destination, project_id=project_id, if_exists='replace')

if __name__ == "__main__":
    gcs_path = 'gs://gdelt_gkg-data/gdelt-gkg-data/20240822.gkg.csv'
    gcs_output_path = 'gs://gdelt_gkg-data/output'
    project_id = 'factoreddatathon2014'
    dataset_id = 'GDELT'
    tsne_table_id = 'dimension_reduction'
    relevant_news_table_id = 'relevant_news'

    run_pipeline(gcs_path, gcs_output_path, project_id, dataset_id, tsne_table_id, relevant_news_table_id)

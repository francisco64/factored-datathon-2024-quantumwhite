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

def process_row(row, selected_tokens, sorted_selected_tokens, columns_to_tokenize):
    urls = row['SOURCEURLS'].split('<UDIV>')
    text_embedding = None
    for url in urls:
        url = url.strip()
        if not url:
            continue
        text = get_text_from_url(url)
        if text:
            text_embedding = get_embedding(text)
            break
    if text_embedding is not None:
        multihot_vector = create_multihot_vector(row, selected_tokens, columns_to_tokenize)
        combined_vector = np.concatenate([multihot_vector, text_embedding])
        return combined_vector
    else:
        logger.warning(f"Skipping row due to no valid URL found.")
        return None

def run_pipeline(gcs_path, gcs_output_path, project_id, dataset_id, tsne_table_id, relevant_news_table_id):
    options = PipelineOptions()

    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.project = project_id
    google_cloud_options.job_name = 'gdelt-relevant-news'
    google_cloud_options.staging_location = f'{gcs_output_path}/staging'
    google_cloud_options.temp_location = f'{gcs_output_path}/temp'
    google_cloud_options.region = 'us-central1'

    options.view_as(StandardOptions).runner = 'DataflowRunner'
    options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(options=options) as p:
        df = pd.read_csv(gcs_path, sep='\t')
        df = df[df['NUMARTS'] >= 3].reset_index(drop=True)

        columns_to_tokenize = ['COUNTS', 'THEMES', 'LOCATIONS', 'PERSONS', 'ORGANIZATIONS']
        token_counts = tokenize_and_count(df, columns_to_tokenize)
        sorted_token_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        token_frequencies = np.array([count for token, count in sorted_token_counts])
        cumulative_freq = np.cumsum(token_frequencies)
        total_freq = cumulative_freq[-1]
        optimal_threshold_idx = np.argmax(cumulative_freq >= 0.8 * total_freq)
        selected_tokens = {token for token, count in sorted_token_counts[:optimal_threshold_idx + 1]}
        sorted_selected_tokens = sorted(selected_tokens)

        combined_embeddings = (
            p
            | 'Create DataFrame Rows' >> beam.Create(df.to_dict('records'))
            | 'Process Rows' >> beam.Map(lambda row: process_row(row, selected_tokens, sorted_selected_tokens, columns_to_tokenize))
            | 'Filter None' >> beam.Filter(lambda x: x is not None)
        )

        X_Norm = np.array(list(combined_embeddings))
        X_Norm = preprocessing.normalize(X_Norm)

        wcss = []
        max_k = 10
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_Norm)
            wcss.append(kmeans.inertia_)
        diff_wcss = np.diff(wcss)
        diff_diff_wcss = np.diff(diff_wcss)
        optimal_k = np.argmin(diff_diff_wcss) + 2

        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans.fit(X_Norm)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_results = tsne.fit_transform(X_Norm)

        df_tsne_clusters = pd.DataFrame({
            'd1': tsne_results[:, 0],
            'd2': tsne_results[:, 1],
            'cluster': kmeans.labels_
        })

        tsne_table_destination = f'{dataset_id}.{tsne_table_id}'
        to_gbq(df_tsne_clusters, destination_table=tsne_table_destination, project_id=project_id, if_exists='replace')

        filtered_df = df.iloc[filtered_indices].reset_index(drop=True)
        filtered_df['cluster_label'] = kmeans.labels_

        closest_samples_per_cluster = []
        for cluster_label in range(optimal_k):
            cluster_points_idx = np.where(kmeans.labels_ == cluster_label)[0]
            cluster_points = X_Norm[cluster_points_idx]
            centroid = kmeans.cluster_centers_[cluster_label]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            closest_idx = cluster_points_idx[np.argmin(distances)]
            closest_samples_per_cluster.append(filtered_df.iloc[closest_idx])

        closest_samples_df = pd.DataFrame(closest_samples_per_cluster)
        closest_to_cluster = closest_samples_df[['cluster_label', 'SOURCEURLS']]

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

import os
import pandas as pd
import re
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
import requests
from bs4 import BeautifulSoup
import tensorflow_hub as hub
from tqdm import tqdm
from pandas_gbq import to_gbq
import vertexai
from vertexai.generative_models import GenerativeModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

print("Loading Universal Sentence Encoder model...")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("Model loaded.")

# Function to extract text from the URL
def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from the HTML, for example from <p> tags
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

# Function to generate the embedding of the text
def get_embedding(text):
    embedding = embed([text]).numpy()
    return np.array(embedding)[0]  # Convert to a numpy array

# Function to tokenize and count tokens in specified columns
def tokenize_and_count(df, columns):
    all_tokens = []
    for col in columns:
        tokens = df[col].dropna().str.split('[#;]').explode().str.strip()
        tokens = tokens[tokens.str.isalpha()]  # Remove non-alphabetic tokens
        all_tokens.extend(tokens)
    return Counter(all_tokens)

# Function to create a multihot vector for a given row
def create_multihot_vector(row, selected_tokens, columns, sorted_selected_tokens):
    tokens = set()
    for col in columns:
        if pd.notna(row[col]):
            col_tokens = set(re.split('[#;]', row[col]))
            tokens.update(col_tokens.intersection(selected_tokens))
    multihot_vector = [1 if token in tokens else 0 for token in sorted_selected_tokens]
    return multihot_vector

def process_file(gcs_path):
    try:
        print("Reading CSV file into DataFrame...")
        df = pd.read_csv(gcs_path, sep='\t')
        print(f"DataFrame shape before filtering: {df.shape}")

        # Filter DataFrame based on NUMARTS
        df = df[df['NUMARTS'] >= 3].reset_index(drop=True)
        print(f"DataFrame shape after filtering NUMARTS >= 3: {df.shape}")
        ##test##
        #df = df.sample(n=100, random_state=42).reset_index(drop=True)
        # Specify columns to tokenize
        columns_to_tokenize = ['COUNTS', 'THEMES', 'LOCATIONS', 'PERSONS', 'ORGANIZATIONS']

        # Tokenize columns and calculate frequencies
        print("Tokenizing columns and calculating token frequencies...")
        token_counts = tokenize_and_count(df, columns_to_tokenize)

        # Sort tokens by frequency
        sorted_token_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        token_frequencies = np.array([count for token, count in sorted_token_counts])

        # Calculate cumulative sum for frequency threshold
        cumulative_freq = np.cumsum(token_frequencies)
        total_freq = cumulative_freq[-1]
        optimal_threshold_idx = np.argmax(cumulative_freq >= 0.8 * total_freq)

        print(f"Optimal threshold index (80% cumulative frequency): {optimal_threshold_idx}")

        # Select tokens for multihot encoding
        selected_tokens = {token for token, count in sorted_token_counts[:optimal_threshold_idx + 1]}
        sorted_selected_tokens = sorted(selected_tokens)  # Sorting to maintain consistent order in multihot vectors

        print(f"Number of selected tokens for multihot encoding: {len(selected_tokens)}")

        combined_embeddings = []
        filtered_indices = []

        print("Processing rows to generate combined embeddings...")

        for index, row in tqdm(df.iterrows(), total=len(df)):
            urls = row['SOURCEURLS']
            text_embedding = None
            if isinstance(urls, str):  # Ensure SOURCEURLS is a string before splitting
                url_list = urls.split('<UDIV>')
                for url in url_list:
                    url = url.strip()
                    if not url:
                        continue
                    text = get_text_from_url(url)
                    if text:  # Only proceed if text was successfully extracted
                        text_embedding = get_embedding(text)
                        break  # Use the first URL that works

            if text_embedding is not None:
                multihot_vector = create_multihot_vector(row, selected_tokens, columns_to_tokenize, sorted_selected_tokens)
                
                # Concatenate multihot vector with text embedding
                combined_vector = np.concatenate([multihot_vector, text_embedding])
                combined_embeddings.append(combined_vector)
                filtered_indices.append(index)
            else:
                print(f"Skipping row {index} due to no valid URL found.")

        combined_embeddings = np.array(combined_embeddings)
        print(f"Combined embeddings shape: {combined_embeddings.shape}")

        # Normalize combined embeddings
        print("Normalizing combined embeddings...")
        X_Norm = preprocessing.normalize(combined_embeddings)
        print("Normalization complete.")

        # Determine optimal number of clusters using the elbow method
        max_k = 10
        wcss = []
        print("Performing KMeans clustering to compute WCSS for different k values...")
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_Norm)
            wcss.append(kmeans.inertia_)
            print(f"Trained KMeans with k = {k}")

        diff_wcss = np.diff(wcss)
        diff_diff_wcss = np.diff(diff_wcss)
        optimal_k = np.argmin(diff_diff_wcss) + 2

        print(f"Fitting KMeans with optimal k = {optimal_k}...")
        kmeans = KMeans(n_clusters=optimal_k, random_state=0)
        kmeans.fit(X_Norm)
        print("KMeans fitting complete.")

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_results = tsne.fit_transform(X_Norm)

        df_tsne_clusters = pd.DataFrame({
            'd1': tsne_results[:, 0],
            'd2': tsne_results[:, 1],
            'cluster': kmeans.labels_
        })

        filtered_df = df.iloc[filtered_indices].reset_index(drop=True)

        project_id = 'factoreddatathon2014'
        dataset_id = 'GDELT'
        table_id = 'dimension_reduction'

        table_destination = f'{dataset_id}.{table_id}'

        to_gbq(
            df_tsne_clusters,
            destination_table=table_destination,
            project_id=project_id,
            if_exists='replace'
        )

        print(f"Data uploaded successfully to {table_destination}")

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

        # Update using .loc to avoid SettingWithCopyWarning
        closest_to_cluster.loc[:, 'extracted_text'] = closest_to_cluster['SOURCEURLS'].apply(
            lambda urls: get_text_from_url(urls.split('<UDIV>')[0].strip()) if isinstance(urls, str) else None
        )

        closest_to_cluster.loc[:, 'summary'] = closest_to_cluster['extracted_text'].apply(
            lambda x: gemini(x) if isinstance(x, str) else None
        )

        # Set the standard deviation for the Gaussian noise
        noise_std_dev = 0.2  # Adjust this value as needed

        for cluster_label in range(optimal_k):
            cluster_points_idx = np.where(kmeans.labels_ == cluster_label)[0]
            cluster_points = X_Norm[cluster_points_idx]
            centroid = kmeans.cluster_centers_[cluster_label]
            
            # Find 10 closest points, each based on a different noisy centroid
            closest_idxs = []
            for _ in range(10):
                # Add Gaussian noise to the centroid for each point
                noisy_centroid = centroid + np.random.normal(0, noise_std_dev, size=centroid.shape)
                
                # Calculate distances from the noisy centroid
                distances = np.linalg.norm(cluster_points - noisy_centroid, axis=1)
                
                # Find the closest point to this noisy centroid that hasn't already been selected
                closest_idx = cluster_points_idx[np.argsort(distances)[0]]
                
                # Append the index to the closest_idxs list if it's not already selected
                if closest_idx not in closest_idxs:
                    closest_idxs.append(closest_idx)
            
            # Add the selected closest points to the list of closest samples
            closest_samples_per_cluster.append(filtered_df.iloc[closest_idxs])

        # Combine all closest samples into a single DataFrame
        closest_samples_df = pd.concat(closest_samples_per_cluster).reset_index(drop=True)

        closest_samples_df['extracted_text'] = closest_samples_df['SOURCEURLS'].apply(
            lambda urls: get_text_from_url(urls.split('<UDIV>')[0].strip()) if isinstance(urls, str) else None
        )

        closest_samples_df['lda_topics'] = ""

        for cluster_label in closest_samples_df['cluster_label'].unique():
            cluster_text = closest_samples_df[closest_samples_df['cluster_label'] == cluster_label]['extracted_text'].dropna().tolist()
            if len(cluster_text) > 0:
                lda_topics = lda_topic_modeling(cluster_text, n_topics=3)
                closest_to_cluster.loc[closest_to_cluster['cluster_label'] == cluster_label, 'lda_topics'] = lda_topics

        table_id = 'relevant_news'
        table_destination = f'{dataset_id}.{table_id}'

        to_gbq(
            closest_to_cluster,
            destination_table=table_destination,
            project_id=project_id,
            if_exists='replace'
        )

        print(f"Data uploaded successfully to {table_destination}")

    except Exception as e:
        print(f"Error processing file {gcs_path}: {e}")

def gemini(txt):
    vertexai.init(project="factoreddatathon2014", location="us-central1")
    model = GenerativeModel("gemini-1.5-flash-001",)
    responses = model.generate_content(
      [
       f'I am trying to get the most important information for a news summary for the next text, be extremely concise and accurate with your words, not more than 3 phrases. If the following text is not worthy of summarizing, put this exact words #nosummary#, do not answer anything:{txt}'
       ],
      generation_config={"max_output_tokens": 100,"temperature": 1,"top_p": 0.95,},
      stream=True,
    )
    response_text = ""
    for response in responses:
        response_text += response.text
    return response_text

def lda_topic_modeling(text_data, n_topics=3):
    count_vectorizer = CountVectorizer(stop_words='english')
    dt_matrix = count_vectorizer.fit_transform(text_data)
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(dt_matrix)
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        topic_keywords = [count_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-n_topics - 1:-1]]
        topics.append(", ".join(topic_keywords))
    return "; ".join(topics)

@app.route('/', methods=['POST'])
def index():
    envelope = request.get_json()

    if not envelope:
        msg = 'No JSON payload received'
        print(f'Error: {msg}')
        return f'Bad Request: {msg}', 400

    try:
        # Extract the relevant data directly from the JSON payload
        if 'bucket' in envelope and 'name' in envelope:
            bucket_name = envelope['bucket']
            file_name = envelope['name']
        else:
            # Handle unexpected payload structure
            msg = 'Invalid message format received'
            print(f'Error: {msg}')
            return f'Bad Request: {msg}', 400

        # Check if the file is a .csv in the correct folder
        specific_folder = "gdelt-gkg-data/"
        if file_name.startswith(specific_folder) and file_name.endswith('.csv'):
            gcs_path = f'gs://{bucket_name}/{file_name}'
            process_file(gcs_path)

    except Exception as e:
        print(f"Error processing the request: {e}")
        return 'Internal Server Error', 500

    return 'OK', 200

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=8080)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:46:01 2024

@author: francisco
"""

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

# Load Universal Sentence Encoder model
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
def create_multihot_vector(row, selected_tokens, columns):
    tokens = set()
    for col in columns:
        if pd.notna(row[col]):
            col_tokens = set(re.split('[#;]', row[col]))
            tokens.update(col_tokens.intersection(selected_tokens))
    multihot_vector = [1 if token in tokens else 0 for token in sorted_selected_tokens]
    return multihot_vector

# Set up environment variables for Google Cloud
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/francisco/Downloads/factoreddatathon2014-915498c6302b.json'
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '1'

# Path to your Google Cloud Storage CSV file
gcs_path = 'gs://gdelt_gkg-data/gdelt-gkg-data/20240822.gkg.csv'

# Read the CSV file directly into a Pandas DataFrame
print("Reading CSV file into DataFrame...")
df = pd.read_csv(gcs_path, sep='\t')
print(f"DataFrame shape before filtering: {df.shape}")

# Filter DataFrame based on NUMARTS
df = df[df['NUMARTS'] >= 2]
print(f"DataFrame shape after filtering NUMARTS >= 2: {df.shape}")

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

# Create multihot vectors and combined embeddings
combined_embeddings = []
filtered_indices = []

print("Processing rows to generate combined embeddings...")

for index, row in df.iterrows():
    # Process URLs and get embedding
    urls = row['SOURCEURLS'].split('<UDIV>')
    text_embedding = None
    for url in urls:
        url = url.strip()
        if not url:
            continue
        text = get_text_from_url(url)
        if text:  # Only proceed if text was successfully extracted
            text_embedding = get_embedding(text)
            break  # Use the first URL that works

    if text_embedding is not None:
        # Create multihot vector
        multihot_vector = create_multihot_vector(row, selected_tokens, columns_to_tokenize)
        
        # Concatenate multihot vector with text embedding
        combined_vector = np.concatenate([multihot_vector, text_embedding])
        combined_embeddings.append(combined_vector)
        filtered_indices.append(index)
    else:
        print(f"Skipping row {index} due to no valid URL found.")

# Check if any embeddings were generated
if not combined_embeddings:
    raise ValueError("No valid embeddings were found. Please check the URLs or data.")

# Create a new DataFrame with only the filtered rows
filtered_df = df.iloc[filtered_indices].reset_index(drop=True)

# Convert to NumPy array
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

# Plotting the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(2, max_k + 1), wcss, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(range(2, max_k + 1))
plt.grid(True)
plt.show()

# Determine optimal k (this can be adjusted based on the elbow plot)
# Here, we'll use the elbow method manually by selecting k where the WCSS starts to decrease linearly
# For automation, more sophisticated methods can be used
# For simplicity, let's select the k with the maximum second derivative (maximum change in slope)
diff_wcss = np.diff(wcss)
diff_diff_wcss = np.diff(diff_wcss)
optimal_k = np.argmin(diff_diff_wcss) + 3  # +2 because diff_wcss has length k-2 and +1 for zero indexing

print(f"Optimal number of clusters (k) determined as: {optimal_k}")

# Fit KMeans with optimal k
print(f"Fitting KMeans with optimal k = {optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_Norm)
print("KMeans fitting complete.")

# Normalize cluster centroids
centroids = preprocessing.normalize(kmeans.cluster_centers_)
print("Centroids normalized.")

# Assign labels to the filtered DataFrame
filtered_df = filtered_df.copy()  # To avoid SettingWithCopyWarning
filtered_df['cluster_label'] = kmeans.labels_

# Find the 4 closest samples to each centroid
closest_samples_df = pd.DataFrame()

print("Identifying the 4 closest samples to each centroid...")

for cluster_idx in range(optimal_k):
    # Get all points in the cluster
    cluster_points_idx = np.where(kmeans.labels_ == cluster_idx)[0]
    cluster_points = X_Norm[cluster_points_idx]
    
    # Calculate the Euclidean distance to the centroid
    distances = np.linalg.norm(cluster_points - centroids[cluster_idx], axis=1)
    
    # Get the indices of the 4 closest points
    closest_points_relative_idx = np.argsort(distances)[:4]
    closest_points_idx = cluster_points_idx[closest_points_relative_idx]
    
    # Extract the corresponding rows from the filtered DataFrame
    closest_samples = filtered_df.iloc[closest_points_idx].copy()
    closest_samples['centroid'] = cluster_idx
    
    # Append to the final DataFrame
    closest_samples_df = pd.concat([closest_samples_df, closest_samples], ignore_index=True)

# Reset index for the final DataFrame
closest_samples_df.reset_index(drop=True, inplace=True)

# Display the final DataFrame with the closest samples
print("Closest samples to each centroid:")
print(closest_samples_df)

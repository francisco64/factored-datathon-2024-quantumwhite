import os
import pandas as pd
import re
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

# Set the environment variable for Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/francisco/Downloads/factoreddatathon2014-915498c6302b.json'
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '1'

gcs_path = 'gs://gdelt_gkg-data/gdelt-gkg-data/20130624.gkg.csv'

# Read the CSV file directly into a Pandas DataFrame
df = pd.read_csv(gcs_path, sep='\t')

def tokenize_and_count(df, columns):
    all_tokens = []
    for col in columns:
        tokens = df[col].dropna().str.split('[#;]').explode().str.strip()
        tokens = tokens[tokens.str.isalpha()]  # Remove numeric values
        all_tokens.extend(tokens)
    return Counter(all_tokens)

# Example usage
columns_to_tokenize = ['COUNTS', 'THEMES', 'LOCATIONS', 'PERSONS', 'ORGANIZATIONS']

token_counts = Counter()

for col in columns_to_tokenize:
    token_counts.update(tokenize_and_count(df, [col]))

# Sort tokens by frequency
sorted_token_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
token_frequencies = np.array([count for token, count in sorted_token_counts])

# Calculate cumulative sum
cumulative_freq = np.cumsum(token_frequencies)
total_freq = cumulative_freq[-1]
optimal_threshold_idx = np.argmax(cumulative_freq >= 0.8 * total_freq)

# Get the tokens that meet the threshold
selected_tokens = {token for token, count in sorted_token_counts[:optimal_threshold_idx + 1]}

def create_multihot_vector(row, selected_tokens, columns):
    tokens = set()
    for col in columns:
        if pd.notna(row[col]):
            col_tokens = set(re.split('[#;]', row[col]))
            tokens.update(col_tokens.intersection(selected_tokens))
    multihot_vector = [1 if token in tokens else 0 for token in selected_tokens]
    return multihot_vector

# Example usage
multihot_vectors = []
for index, row in df.iterrows():
    vector = create_multihot_vector(row, selected_tokens, columns_to_tokenize)
    multihot_vectors.append(vector)

# Convert to sparse matrix to save memory
bow = np.array(multihot_vectors)

# import requests
# from bs4 import BeautifulSoup
# import tensorflow_hub as hub

# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# # Function to extract text from the URL
# def get_text_from_url(url):
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()  # Raise an error for bad status codes
#         soup = BeautifulSoup(response.content, 'html.parser')

#         # Extract text from the HTML, for example from <p> tags
#         paragraphs = soup.find_all('p')
#         text = ' '.join([para.get_text() for para in paragraphs])
#         return text
#     except requests.RequestException as e:
#         print(f"Error fetching {url}: {e}")
#         return None

# # Function to generate the embedding of the text
# def get_embedding(text):
#     embedding = embed([text]).numpy()
#     return np.array(embedding)[0]  # Convert to a numpy array

# # Example usage: Process each URL in the DataFrame
# embeddings = []
# for url in df['SOURCEURLS']:  
#     url=url.split('<UDIV>')[0]# Assuming 'SOURCEURLS' is the column with URLs
#     if pd.notna(url):  # Check if the URL is not NaN
#         text = get_text_from_url(url)
#         if text:  # Only proceed if text was successfully extracted
#             embedding = get_embedding(text)
#             embeddings.append(embedding)
#         else:
#             continue
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt



# Compute cosine distance matrix
cosine_dist_matrix = cosine_distances(bow[0:10000,:])

# Function to find the optimal number of clusters using the Elbow Method
def find_optimal_k_elbow(X, max_k=10):
    wcss = []  # Within-cluster sum of squares (WCSS)
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        print("trained kmeans with k equals {}".format(k))
    
    # Plotting the elbow curve
    plt.plot(range(2, max_k+1), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Inertia)')
    plt.show()


# Find the optimal k using the elbow method
#find_optimal_k_elbow(cosine_dist_matrix)


kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(cosine_dist_matrix)

# # Find the optimal k using the silhouette score method
# optimal_k = find_optimal_k_silhouette(cosine_dist_matrix)
# print(f'Optimal number of clusters (k) according to Silhouette Score: {optimal_k}')

# # Now perform KMeans clustering with the optimal number of clusters
# kmeans = KMeans(n_clusters=optimal_k, random_state=42)
# kmeans.fit(cosine_dist_matrix)

# # Get the cluster labels
# labels = kmeans.labels_
# print(labels)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Convert the cosine distance matrix to a 2D space using t-SNE
tsne = TSNE(n_components=2, metric='cosine', random_state=42,perplexity=kmeans.n_clusters - 1)
tsne_results = tsne.fit_transform(bow[0:10000])

# Plot the t-SNE results along with the cluster centroids
plt.figure(figsize=(10, 7))

# Plot the t-SNE results
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.7)

# Plot the centroids
centroids_tsne = tsne.fit_transform(kmeans.cluster_centers_)
plt.scatter(centroids_tsne[:, 0], centroids_tsne[:, 1], s=300, c='red', label='Centroids')

plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE with K-Means Centroids')
plt.legend()
plt.show()

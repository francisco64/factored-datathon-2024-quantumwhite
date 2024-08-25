# K-Relevant News System: 
## Introduction

The K-Relevant News System is an advanced pipeline designed to retrieve and summarize the most relevant news articles by leveraging machine learning techniques like K-Means clustering and topic modeling. This system processes feature vectors derived from both multihot encoding of tokens (extracted from themes, organizations, and persons in the GKG schema) and dense embeddings generated from the text of HTML content via the Google Universal Sentence Encoder.

This document provides a comprehensive explanation of the methodologies employed, the experiments conducted, and the conclusions drawn from this system, along with the deployment architecture on Google Cloud.

## Methodology

### 1. Token Selection and Multihot Encoding

The foundation of the K-Relevant News System lies in efficiently encoding the vast amounts of textual data available. This starts with selecting and encoding tokens from themes, organizations, and persons in the GKG schema.

#### Key Experiment: Token Filtering

- **Challenge:** With potentially thousands of tokens available, it was essential to filter out the noise and focus on the most impactful ones.
- **Solution:** By analyzing token frequencies, it was found that around 50 tokens accounted for 80% of the occurrences. These tokens were selected for the multihot encoding, ensuring that the most relevant and frequent features were prioritized.
- **Conclusion:** Reducing the token set to these key tokens enhanced the efficiency of the multihot encoding process and allowed for better clustering outcomes by focusing on the most influential features.

### 2. Embedding with Google Universal Sentence Encoder

In parallel to the multihot encoding, text extracted from HTML content is processed using the Google Universal Sentence Encoder to obtain dense embeddings. This encoder generates a 512-dimensional vector for each document, capturing semantic nuances in the text.

- **Importance:** These embeddings provide a rich representation of the text, which, when combined with the sparse binary vectors from multihot encoding, form a comprehensive feature set for clustering.

### 3. Clustering with K-Means and GMM

Once the feature vectors are obtained, clustering is performed to group similar news articles together.

#### Key Experiments and Findings:

- **Optimal K using the Second Derivative of Inertia:**
  - **Challenge:** Determining the optimal number of clusters (K) is crucial for meaningful clustering. 
  - **Solution:** The elbow method was employed, where the second derivative of the inertia curve was calculated to pinpoint the "elbow" point. This method allows for an automated and accurate determination of the optimal K.
  - **Conclusion:** The use of the second derivative of inertia to determine the optimal K ensured that the clusters were both meaningful and well-separated.

- **K-Means vs. GMM:**
  - **Experiment:** Both K-Means and Gaussian Mixture Models (GMM) were implemented. Due to the normalization of vectors, the cosine distance in K-Means and the Euclidean distance in GMM were effectively equivalent, allowing the models to be converted and compared.
  - **Conclusion:** The similarity in distance metrics between K-Means and GMM provided flexibility in model choice, without sacrificing clustering performance.

- **Centroid Analysis with NUMARTS:**
  - **Experiment:** Initial clustering attempts using samples with maximum `NUMARTS` (number of articles) resulted in centroids that poorly represented the data. These centroids were often skewed by outliers and noise.
  - **Improvement:** By selecting samples that were closest to the centroids, a significant improvement in cluster quality was observed. These centroids more accurately represented the core of each cluster.
  - **Conclusion:** Selecting centroid-based samples led to better cluster representation, which in turn improved the relevance of the retrieved news articles.

- **Enhancing Diversity with Gaussian Noise:**
  - **Challenge:** When retrieving the top 10 samples closest to each centroid, it was found that many samples were repeated, reducing the diversity of the results.
  - **Solution:** Gaussian noise was added to the centroids before retrieving the closest samples. This approach introduced slight variations, ensuring that the retrieved samples were diverse while still being close to the cluster centers.
  - **Conclusion:** Adding Gaussian noise effectively increased the diversity of the retrieved news articles, making the clustering results more comprehensive.

### 4. Topic Modeling with LDA

After clustering, the selected samples (those closest to the noisy centroids) were further analyzed using Latent Dirichlet Allocation (LDA) to identify the underlying topics within each cluster.

- **Purpose:** LDA was used to provide deeper insights into the composition of each cluster by revealing the key topics that define the grouped news articles.

### 5. Summarization with Gemini

The final step in the content analysis pipeline involves summarizing the texts of the selected samples using Gemini. This tool produces concise summaries of the most relevant news articles in each cluster.

- **Outcome:** The summaries generated by Gemini offer a quick and informative overview of the key points in each cluster, facilitating easy consumption of large volumes of information.

## Deployment on Google Cloud

### System Architecture

The K-Relevant News System is deployed on Google Cloud to leverage its scalable and powerful infrastructure.

#### Key Components:

- **Google Cloud Functions:** Triggers the pipeline upon detecting changes in a Google Cloud Storage bucket. This serverless function initiates the retrieval and processing of news articles.
- **Google Compute Engine:** Handles the intensive computation tasks, including processing embeddings with the Google Universal Sentence Encoder and performing clustering. The choice of Compute Engine ensures that the system can scale as needed to handle large datasets.
- **BigQuery:** Stores the processed data, including the clustered news articles and topic modeling results. BigQuery’s powerful querying capabilities allow for efficient data retrieval and analysis.

### Conclusion

The K-Relevant News System combines advanced machine learning techniques with robust cloud infrastructure to deliver a powerful tool for news retrieval and analysis. The methodology, including the optimization of token selection, the precise determination of the optimal number of clusters, and the innovative use of Gaussian noise to enhance sample diversity, ensures that the system provides accurate and relevant results. The deployment on Google Cloud further ensures that the system is scalable, reliable, and ready for production use.

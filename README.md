# factored_competition_2024
 
# GDELT Insight Hub Dashboard

## Overview
The GDELT Insight Hub Dashboard is an end-to-end data processing and visualization platform designed to analyze global events on a daily basis using the GDELT dataset. This repository contains the code, Docker files, and configurations necessary to deploy the entire pipeline on Google Cloud Platform (GCP), offering a powerful tool for monitoring, analyzing, and visualizing global events.

## Architecture

![Architecture Diagram](https://storage.googleapis.com/images_superset/end-to-end-architecture.png)

The architecture is designed for scalability and flexibility, using the following key components:

- **Dataflow Pipeline:**
  - Ingests both full datasets and daily batches from the GDELT endpoint, processing the data and storing it in Google Cloud Storage.
  
- **Cloud Storage & Cloud Functions:**
  - Cloud Storage acts as a central repository for processed data. Cloud Functions are triggered by new data uploads to initiate further processing in Compute Engine.

- **Compute Engine:**
  - Runs advanced data processing tasks, including KMeans clustering and LDA topic modeling. This is where the core analysis happens, utilizing TensorFlow Hub’s Universal Sentence Encoder for embeddings.

- **BigQuery:**
  - Serves as the primary data warehouse, storing processed data and powering the SQL queries that feed into the dashboard.

- **Apache Superset:**
  - A powerful, open-source data exploration and visualization tool that connects directly to BigQuery, providing a dynamic, interactive dashboard interface for users.

- **Docker & Container Registry:**
  - Automates the deployment of data processing tasks using Docker containers. These containers are stored in Google Container Registry for easy management and scalability.

- **Gemini AI Integration:**
  - Utilizes Google Gemini AI for summarizing key insights from the news data, enhancing user comprehension and decision-making.

## Features

- **Daily Monitoring:** 
  - Continuously ingest and analyze global event data as it happens, ensuring up-to-date insights.
  
- **Advanced Analytics:**
  - Perform clustering, topic modeling, and other sophisticated analyses to uncover patterns and trends.

- **Interactive Visualizations:**
  - Utilize Apache Superset for rich, interactive dashboards that help you explore and understand the data.

## Getting Started

### Prerequisites

- **Google Cloud Platform Account:** Set up a GCP project to deploy the components.
- **Docker:** Ensure Docker is installed for containerizing the components.
- **Python 3.x:** Required for running the scripts locally if needed.
- **Apache Superset:** Install and configure Superset for dashboard visualization.

### Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/gdelt-insight-hub.git
    cd gdelt-insight-hub
    ```

2. **Build and Push Docker Containers:**
    ```bash
    docker build -t gcr.io/your-project-id/gdelt-dataflow .
    docker push gcr.io/your-project-id/gdelt-dataflow
    ```

3. **Deploy Dataflow Pipelines:**
    - Use the provided Dataflow templates and configurations to set up the ingestion pipelines.

4. **Configure and Deploy Compute Engine Instances:**
    - Deploy the Compute Engine instance and install necessary packages, including TensorFlow Hub.

5. **Set Up BigQuery and Superset:**
    - Configure your BigQuery datasets and connect them to Apache Superset for visualization.

6. **Set Up Alerts and Notifications:**
    - Configure custom alerts based on your specific needs using Cloud Functions or BigQuery.

## Usage

Once deployed, the Insight Hub Dashboard provides a real-time, interactive interface to explore global events, analyze trends, and gain actionable insights. The dashboard is customizable and can be tailored to specific regions, industries, or types of events.

## Contact

For more information, please contact:
- **Francisco Reales**
- **Email:** reales.francisco1@gmail.com
- **LinkedIn:** [franciscoreales](https://www.linkedin.com/in/franciscoreales/)



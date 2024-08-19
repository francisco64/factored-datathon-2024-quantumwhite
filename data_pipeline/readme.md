# GDELT Data Pipeline Documentation

## Overview

This documentation outlines the creation and execution of a data pipeline designed to process GDELT (Global Database of Events, Language, and Tone) data, specifically event or GKG (Global Knowledge Graph) data, and load it into Google BigQuery. The pipeline leverages Apache Beam and Google Cloud Dataflow for efficient, scalable data processing and ingestion tasks, facilitating both the processing of thousands of URLs and daily scheduled jobs.

## Methodology

### Apache Beam

Apache Beam is a powerful, unified model for both batch and streaming data processing. It simplifies the process of defining complex data pipelines and provides a level of abstraction that allows for execution across various environments. In this project, Apache Beam is used to automate the following tasks:

1. **Fetching Data:** The pipeline retrieves .zip files from the GDELT website.
2. **Extracting Data:** The .zip files are decompressed to access the underlying CSV files.
3. **Uploading to GCS:** The extracted CSV files are uploaded to Google Cloud Storage (GCS).
4. **Loading to BigQuery:** The data is loaded from GCS into BigQuery, where it can be queried and analyzed.

### Google Cloud Dataflow

Google Cloud Dataflow is a fully managed service that runs Apache Beam pipelines. It offers the following advantages:

- **Scalability:** Dataflow automatically scales the number of worker nodes based on data volume, ensuring that large datasets are processed efficiently.
- **Managed Infrastructure:** Dataflow handles all infrastructure needs, including resource allocation, job scheduling, and monitoring, which reduces operational overhead.
- **Flexibility:** Supports both batch and streaming data processing, making it versatile for a wide range of use cases.

### Comparison with Apache Spark

While Apache Spark is another popular data processing framework, Apache Beam combined with Dataflow offers several key advantages:

- **Unified Processing Model:** Beam's model supports both batch and streaming data processing without requiring changes to the pipeline code.
- **Managed Environment:** Dataflow’s fully managed service simplifies the operational complexity compared to managing Spark clusters.
- **Dynamic Resource Scaling:** Dataflow’s autoscaling adjusts resources dynamically based on workload, optimizing both performance and cost.

### General Architecture

The architecture consists of the following key components:

- **Pipeline Definition:** The data pipeline is defined using Apache Beam in a Python script. This script outlines the process of fetching, extracting, uploading, and loading data.
- **Docker Image Creation:** The pipeline script and its dependencies are packaged into a Docker image, which is then uploaded to Google Container Registry (GCR).
- **Flex Template Creation:** A Flex Template is created from the Docker image. This template allows for the easy scheduling and execution of data processing jobs by specifying runtime parameters.
- **Job Scheduling:** The Flex Template is used to schedule and run Dataflow jobs, either on demand or as part of a recurring batch process.

### Flex Template

Flex Templates in Google Dataflow allow for the packaging and deployment of Apache Beam pipelines in a reusable format. This enables the easy customization of pipeline execution through runtime parameters, without the need to modify the underlying code.

### Performance Metrics

Below is an example of the expected performance metrics for processing different types of GDELT data:

| Data Type       | Average Processing Time | Average Data Size | Notes                             |
|-----------------|-------------------------|-------------------|-----------------------------------|
| Events Data     | 3 hours                 | 200 GB            | Batch processing of daily data    |
| GKG Data        | 2.5 hours               | 150 GB            | Batch processing of daily data    |

## Step-by-Step Guide

### Direct Job Execution

You can run the pipeline directly using the command outlined in the Python script provided in the repository. This method is ideal for debugging or one-time executions.

```bash
python gdelt2bq.py \
  --table_name 'gkg-data' \
  --region 'southamerica-west1' \
  --project_id 'factoreddatathon2014' \
  --url_of_index 'http://data.gdeltproject.org/gkg/index.html' \
  --in_url '.zip' \
  --not_in_url 'counts' \
  --schema gkg \
  --day_before_condition 1
```

### Docker Image Creation

The data pipeline script, along with all necessary dependencies, is packaged into a Docker image. This image ensures that the pipeline can be executed in a consistent environment, regardless of the underlying infrastructure.

```bash
# Use the official Dataflow base image
FROM gcr.io/dataflow-templates-base/python3-template-launcher-base

# Install required Python packages
COPY requirements.txt /app/requirements.txt
COPY gdelt2bq.py /app/gdelt2bq.py
COPY gdelt_schemas /app/gdelt_schemas

# Set the environment variable for the Dataflow template launcher
ENV FLEX_TEMPLATE_PYTHON_PY_FILE="/app/gdelt2bq.py"

RUN apt-get update
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PIP_NO_DEPS=True
```
docker image creation and registry
```bash
docker build -t gcr.io/factoreddatathon2014/gdelt2bq:latest .
docker push gcr.io/factoreddatathon2014/gdelt2bq:latest
```

### Flex Template Creation

A Flex Template is created from the Docker image. This template allows the pipeline to be easily deployed and executed with different parameters, facilitating multiple use cases from a single template.

template_spec.json

```bash
{
  "name": "gdelt_data_pipeline_flex_template",
  "description": "A Flex Template for processing GDELT event or GKG data and loading it into BigQuery.",
  "parameters": [
    {
      "name": "table_name",
      "label": "Table Name",
      "helpText": "The name of the GDELT table (e.g., 'events' or 'gkg').",
      "isOptional": false
    },
    {
      "name": "region",
      "label": "Region",
      "helpText": "The GCP region where the Dataflow job will run.",
      "isOptional": false
    },
    {
      "name": "project_id",
      "label": "Project ID",
      "helpText": "The ID of the GCP project.",
      "isOptional": false
    },
    {
      "name": "url_of_index",
      "label": "URL of Index",
      "helpText": "The URL of the index.html containing the file links to process.",
      "isOptional": false
    },
    {
      "name": "in_url",
      "label": "Include in URL",
      "helpText": "A list of terms that must be present in the URL to be processed.",
      "isOptional": false
    },
    {
      "name": "not_in_url",
      "label": "Exclude from URL",
      "helpText": "A list of terms that should be excluded from the URLs being processed.",
      "isOptional": true
    },
    {
      "name": "schema",
      "label": "BigQuery Schema",
      "helpText": "Specify the schema to use for the BigQuery table (e.g., 'events' or 'gkg').",
      "isOptional": false
    },
    {
      "name": "day_before_condition",
      "label": "Day Before Condition",
      "helpText": "Set to 1 to filter URLs by the day before the current date, 0 otherwise.",
      "isOptional": false
    }
  ],
  "sdkInfo": {
    "language": "PYTHON"
  },
  "defaultEnvironment": {}
}
```

command to create the flex template:

```bash
gcloud dataflow flex-template build gs://data_pipeline_scripts/templates/gdelt2bq-template \
  --image "gcr.io/factoreddatathon2014/gdelt2bq:latest" \
  --sdk-language "PYTHON" \
  --metadata-file "template_spec.json"
```


### Job Scheduling with Flex Template

Using the Flex Template, jobs can be scheduled and run through the Google Cloud Dataflow UI or via command-line tools. This makes it easy to automate the data pipeline and ensure timely data processing.

```bash
gcloud dataflow flex-template run "gdelt2bq-job" \
  --template-file-gcs-location "gs://data_pipeline_scripts/templates/gdelt2bq-template" \
  --parameters table_name='gkg-data',region='southamerica-west1',project_id='factoreddatathon2014',url_of_index='http://data.gdeltproject.org/gkg/index.html',in_url='.zip',not_in_url='counts',schema='gkg',day_before_condition=1 \
  --region='southamerica-west1'

```
## Conclusion

This data pipeline showcases the power and flexibility of Apache Beam and Google Cloud Dataflow for large-scale data processing tasks. By combining these technologies with Docker and Flex Templates, the pipeline becomes highly modular, scalable, and easy to manage. This approach significantly reduces the complexity and operational overhead involved in processing and ingesting large datasets into BigQuery.

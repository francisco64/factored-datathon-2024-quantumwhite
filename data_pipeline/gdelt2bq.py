#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:28:56 2024

@author: franciscoreales
"""
import google.api_core.exceptions 
import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions, SetupOptions, WorkerOptions
import requests
import zipfile
import io
from google.cloud import storage, bigquery
import time
from datetime import datetime, timedelta
import argparse
import pytz
from gdelt_schemas.gkg_schema import schema as gkg_schema
from gdelt_schemas.events_schema import schema as events_schema

# Function to download and extract zip files
def download_and_extract_zip(url, logger):
    try:
        logger.info(f"Downloading {url}...")
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for filename in z.namelist():
                if filename.endswith('.CSV') or filename.endswith('.csv'):
                    logger.info(f"Extracting {filename}...")
                    with z.open(filename) as f:
                        return filename, f.read()
    except Exception as e:
        logger.error(f"Error downloading or extracting zip file from {url}: {str(e)}")
        return None

# Function to upload CSV content to GCS
def upload_to_gcs(filename_content, gcs_bucket, gcs_path, logger):
    try:
        if filename_content is not None:
            filename, content = filename_content
            storage_client = storage.Client()
            bucket = storage_client.bucket(gcs_bucket)
            blob = bucket.blob(f"{gcs_path}{filename}")
            blob.upload_from_string(content)
            logger.info(f"Uploaded {filename} to GCS.")
            return f"gs://{gcs_bucket}/{gcs_path}{filename}"
    except Exception as e:
        logger.error(f"Error uploading {filename_content[0]} to GCS: {str(e)}")
        return None

# Function to load data into BigQuery with retry logic
# def load_to_bigquery(gcs_uri, bq_dataset, bq_table, table_name, gcs_bucket, logger, schema):
#     try:
#         if gcs_uri is not None:
#             bq_client = bigquery.Client()
#             dataset_ref = bq_client.dataset(bq_dataset)
#             table_ref = dataset_ref.table(bq_table)
            
#             schema_table = events_schema if schema == 'events' else gkg_schema
            
#             try:
#                 bq_client.get_table(table_ref)
#                 logger.info(f"Table {bq_table} already exists.")
#             except Exception as e:
#                 logger.info(f"Table {bq_table} does not exist. Creating table: {str(e)}")
#                 table = bigquery.Table(table_ref, schema=schema_table)
#                 table = bq_client.create_table(table)
#                 logger.info(f"Table {bq_table} created.")

#             job_config = bigquery.LoadJobConfig(
#                 source_format=bigquery.SourceFormat.CSV,
#                 skip_leading_rows=0 if schema == 'events' else 1,  # gkg table has column names on the first row
#                 schema=schema_table,
#                 field_delimiter='\t'
#             )

#             for attempt in range(5):  # Retry up to 5 times
#                 try:
#                     load_job = bq_client.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)
#                     load_job.result()  # Waits for the job to complete
#                     logger.info(f"Loaded {gcs_uri} into BigQuery table {bq_table}")
#                     break  # Break the loop if successful
#                 except Exception as e:
#                     logger.error(f"Error loading {gcs_uri} into BigQuery on attempt {attempt + 1}: {str(e)}")
#                     time.sleep(10)  # Wait before retrying
#     except Exception as e:
#         logger.error(f"Error in load_to_bigquery function: {str(e)}")
def load_to_bigquery(gcs_uri, bq_dataset, bq_table, table_name, gcs_bucket, logger, schema):
    if gcs_uri is not None:
        bq_client = bigquery.Client()
        dataset_ref = bq_client.dataset(bq_dataset)
        table_ref = dataset_ref.table(bq_table)
        
        schema_table = events_schema if schema == 'events' else gkg_schema
        
        try:
            bq_client.get_table(table_ref)
            logger.info(f"Table {bq_table} already exists.")
        except Exception:
            table = bigquery.Table(table_ref, schema=schema_table)
            table = bq_client.create_table(table)
            logger.info(f"Table {bq_table} created.")

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=0 if schema == 'events' else 1,
            schema=schema_table,
            field_delimiter='\t'
        )

        for attempt in range(5):  # Retry up to 5 times with delay
            try:
                load_job = bq_client.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)
                load_job.result()  # Waits for the job to complete
                logger.info(f"Loaded {gcs_uri} into BigQuery table {bq_table}")
                break  # Break the loop if successful
            except google.api_core.exceptions.Forbidden as e:
                if 'rateLimitExceeded' in str(e):
                    logger.warning(f"Rate limit exceeded. Attempt {attempt + 1}. Retrying...")
                    time.sleep(60)  # Delay before retrying
                else:
                    logger.error(f"Error loading {gcs_uri} into BigQuery: {str(e)}")
                    break  # Stop retrying on other errors
            except Exception as e:
                logger.error(f"Error loading {gcs_uri} into BigQuery: {str(e)}")
                break  # Stop retrying on other errors

# Function to get the list of URLs to process
def generate_file_list(url_of_index, in_url, not_in_url, logger, day_before_condition):
    try:
        response = requests.get(url_of_index)
        file_urls = []
        
        if day_before_condition == 1:
            colombian_time_zone = pytz.timezone('America/Bogota')
            current_date = datetime.now(colombian_time_zone)
            previous_day = current_date - timedelta(days=1)
            formatted_date = previous_day.strftime('%Y%m%d')
            in_url.append(formatted_date)

        if response.status_code == 200:
            lines = response.text.splitlines()
            for line in lines:
                if all(term in line for term in in_url) and not any(term in line for term in not_in_url):
                    url = url_of_index.rsplit('/', 1)[0] + '/' + line.split('"')[1]
                    file_urls.append(url)
        return file_urls
    except Exception as e:
        logger.error(f"Error generating file list from {url_of_index}: {str(e)}")
        return []

# Apache Beam pipeline
def run_pipeline(gcs_bucket, 
                 gcs_path, 
                 bq_dataset, 
                 bq_table, 
                 region, 
                 project_id, 
                 urls, 
                 table_name, 
                 logger,
                 schema):
    try:
        options = PipelineOptions()

        google_cloud_options = options.view_as(GoogleCloudOptions)
        google_cloud_options.project = project_id
        google_cloud_options.job_name = f'gdelt-{table_name}-dataflow-job'
        google_cloud_options.staging_location = f'gs://{gcs_bucket}/staging'
        google_cloud_options.temp_location = f'gs://{gcs_bucket}/temp'
        google_cloud_options.region = region
        options.view_as(SetupOptions).save_main_session = True

        options.view_as(StandardOptions).runner = 'DataflowRunner'

        worker_options = options.view_as(WorkerOptions)
        worker_options.autoscaling_algorithm = 'THROUGHPUT_BASED'
        worker_options.max_num_workers = 50
        worker_options.num_workers = 5

        with beam.Pipeline(options=options) as p:
            gcs_uris = (
                p
                | 'Download and Extract Zip' >> beam.Create(urls)
                | 'Extract and Upload to GCS' >> beam.Map(download_and_extract_zip, logger=logger)
                | 'Filter None' >> beam.Filter(lambda x: x is not None)
                | 'Upload to GCS' >> beam.Map(upload_to_gcs, gcs_bucket=gcs_bucket, gcs_path=gcs_path, logger=logger)
                | 'Filter GCS None' >> beam.Filter(lambda x: x is not None)
            )
            gcs_uris | 'Load into BigQuery' >> beam.Map(load_to_bigquery, 
                                                        bq_dataset=bq_dataset, 
                                                        bq_table=bq_table, 
                                                        table_name=table_name,  
                                                        gcs_bucket=gcs_bucket, 
                                                        logger=logger,
                                                        schema=schema)
    except Exception as e:
        logger.error(f"Error in run_pipeline function: {str(e)}")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--table_name', required=True, help='Table name for GDELT (e.g., events or gkg)')
        parser.add_argument('--region', required=True, help='GCP region')
        parser.add_argument('--project_id', required=True, help='GCP Project ID')
        parser.add_argument('--url_of_index', required=True, help='URL of the index.html containing the file links')
        parser.add_argument('--in_url', nargs='+', required=True, help='Terms that should be in the URL to download')
        parser.add_argument('--not_in_url', nargs='+', required=False, default=[], help='Terms to exclude in URLs')
        parser.add_argument('--schema', required=True, help='Schema of the BigQuery table: events or gkg')
        parser.add_argument('--day_before_condition', type=int, required=True, help='1 or 0 if only read URLs that contain the date of the day before running this command - Colombian time')
        
        args, pipeline_args = parser.parse_known_args()

        gcs_bucket = f"gdelt_{args.table_name}"
        gcs_path = f"gdelt-{args.table_name}/"
        bq_table = args.table_name
        bq_dataset = "GDELT"

        logger = logging.getLogger(__name__)
        if not logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)

        urls = generate_file_list(args.url_of_index, args.in_url, args.not_in_url, logger, args.day_before_condition)
        run_pipeline(gcs_bucket, 
                     gcs_path, 
                     bq_dataset, 
                     bq_table, 
                     args.region, 
                     args.project_id, 
                     urls, 
                     args.table_name, 
                     logger,
                     args.schema)
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
import os
import requests
import json
from google.cloud import storage

def trigger_cloud_run(event, context):
    """Background Cloud Function to be triggered by Cloud Storage.
       This function makes a POST request to the Cloud Run service.
    Args:
        event (dict): The dictionary with data specific to this type of event.
        context (google.cloud.functions.Context): Metadata of triggering event.
    """

    file_data = {
        "bucket": event['bucket'],
        "name": event['name']
    }

    # Define the Cloud Run service URL
    cloud_run_url = "http://34.148.93.109:8080/"

    # Debugging: print the cloud_run_url to the logs
    print(f"CLOUD_RUN_URL: {cloud_run_url}")

    if not cloud_run_url:
        raise ValueError("CLOUD_RUN_URL environment variable is not set.")

    # Send the file_data directly as JSON payload
    response = requests.post(cloud_run_url, json=file_data)

    # Log the response
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text}")

    if response.status_code != 200:
        raise RuntimeError(f"Failed to trigger Cloud Run service: {response.text}")
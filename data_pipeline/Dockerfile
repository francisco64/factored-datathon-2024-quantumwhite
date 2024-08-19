# Use the official Dataflow base image
FROM gcr.io/dataflow-templates-base/python3-template-launcher-base

# Install required Python packages
COPY requirements.txt /app/requirements.txt
# Copy the Python script to the Docker image
COPY gdelt2bq.py /app/gdelt2bq.py
COPY gdelt_schemas /app/gdelt_schemas
# Set the environment variable for the Dataflow template launcher
ENV FLEX_TEMPLATE_PYTHON_PY_FILE="/app/gdelt2bq.py"
#ENV FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE="/app/requirements.txt"


RUN apt-get update
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

#RUN pip install 'apache-beam[gcp]==2.58.0'
#RUN pip install -U -r /app/requirements.txt

ENV PIP_NO_DEPS=True
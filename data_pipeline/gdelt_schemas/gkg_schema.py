#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 19:14:23 2024

@author: franciscoreales
"""

from google.cloud import bigquery

schema = [
    bigquery.SchemaField("DATE", "STRING"),
    bigquery.SchemaField("NUMARTS", "INTEGER"),
    bigquery.SchemaField("COUNTS", "STRING"),
    bigquery.SchemaField("THEMES", "STRING"),
    bigquery.SchemaField("LOCATIONS", "STRING"),
    bigquery.SchemaField("PERSONS", "STRING"),
    bigquery.SchemaField("ORGANIZATIONS", "STRING"),
    bigquery.SchemaField("TONE", "STRING"),
    bigquery.SchemaField("CAMEOEVENTIDS", "STRING"),
    bigquery.SchemaField("SOURCES", "STRING"),
    bigquery.SchemaField("SOURCEURLS", "STRING")
]

# services/bq.py
"""
BigQuery helpers (ADC-based)

- Uses Application Default Credentials (ADC). Make sure you ran:
    gcloud auth application-default login
- Client is cached for reuse across queries.

Exports:
  - get_bq_client() -> bigquery.Client
  - run_query(sql: str)
  - run_query_to_df(sql: str) -> pandas.DataFrame
"""

from __future__ import annotations

from google.cloud import bigquery
import pandas as pd

_bq_client = None


def get_bq_client() -> bigquery.Client:
    """
    Return a cached BigQuery client using ADC (Application Default Credentials).
    Region/project are inferred from your environment/ADC context.
    """
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client()
    return _bq_client


def run_query(sql: str):
    """
    Execute a SQL string and block until completion.
    Returns the completed QueryJob (useful for metadata/logs if needed).
    """
    client = get_bq_client()
    job = client.query(sql)
    job.result()  # block until done
    return job


def run_query_to_df(sql: str) -> pd.DataFrame:
    """
    Execute a SQL string and return results as a pandas DataFrame.
    """
    client = get_bq_client()
    return client.query(sql).to_dataframe()

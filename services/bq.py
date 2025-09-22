from google.cloud import bigquery
import pandas as pd

_bq_client = None

def get_bq_client() -> bigquery.Client:
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client()  
    return _bq_client


def run_query(sql: str):
    client = get_bq_client()
    job = client.query(sql)
    job.result()  # block until done
    return job

def run_query_to_df(sql: str) -> pd.DataFrame:
    client = get_bq_client()
    return client.query(sql).to_dataframe()




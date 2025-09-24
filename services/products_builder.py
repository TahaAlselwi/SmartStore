# services/products_builder.py
"""
Build product embeddings (text + image) inside BigQuery.

Pipeline:
  1) Text embeddings:
     - Adds `text_embedding ARRAY<FLOAT64>` if missing.
     - Backfills using ML.GENERATE_EMBEDDING over (title + description).

  2) Image embeddings:
     - Adds `uri STRING` if missing and fills it from GCS prefix + image filename.
     - Adds `img_embedding ARRAY<FLOAT64>` if missing.
     - Backfills using ML.GENERATE_EMBEDDING over a signed-access URL from OBJ.*.

Notes:
- Ensure the BigQuery remote connection used in OBJ.* has read access to your GCS bucket.
- Regions for dataset, models, and connection should match (e.g., all in US).
"""

from services.bq import run_query
from config import (
    PRODUCTS_TABLE_ID,
    GCS_Images_URI,
    OBJ_CONNECTION_ID,
    TEXT_Embedding_MODEL_ID,
    MM_Embedding_MODEL_ID,
)


def _normalize_gcs_prefix(prefix: str) -> str:
    """Ensure the GCS prefix ends with a single trailing slash."""
    if not prefix:
        return prefix
    if not prefix.endswith("/"):
        return prefix + "/"
    return prefix


def build_text_embeddings(
    table_full_id: str = PRODUCTS_TABLE_ID,
    text_model_full_id: str = TEXT_Embedding_MODEL_ID,
    id_col: str = "product_id",
    title_col: str = "title",
    desc_col: str = "description",
    embedding_col: str = "text_embedding",
):
    """
    Create and backfill text embeddings for products.

    Args:
      table_full_id: Fully qualified table ID, e.g. "project.dataset.products".
      text_model_full_id: Remote model ID for ML.GENERATE_EMBEDDING (text).
      id_col/title_col/desc_col: Column names in products table.
      embedding_col: Destination column for the embedding.
    """
    # 1) Add embedding column if missing
    sql_add_col = f"""
    ALTER TABLE `{table_full_id}`
    ADD COLUMN IF NOT EXISTS {embedding_col} ARRAY<FLOAT64>;
    """
    run_query(sql_add_col)

    # 2) Backfill embeddings
    #    We concatenate title + description for richer signals.
    sql_backfill = f"""
    UPDATE `{table_full_id}` AS t
    SET t.{embedding_col} = s.embedding
    FROM (
      SELECT
        {id_col},
        ml_generate_embedding_result AS embedding
      FROM ML.GENERATE_EMBEDDING(
        MODEL `{text_model_full_id}`,
        (
          SELECT
            {id_col},
            CONCAT(COALESCE({title_col}, ''), ' ', COALESCE({desc_col}, '')) AS content
          FROM `{table_full_id}`
        ),
        STRUCT(TRUE AS flatten_json_output)
      )
    ) AS s
    WHERE t.{id_col} = s.{id_col};
    """
    run_query(sql_backfill)


def build_image_embeddings(
    table_full_id: str = PRODUCTS_TABLE_ID,          # e.g., "project.dataset.products"
    gcs_prefix: str = GCS_Images_URI,               # e.g., "gs://my-bucket/images/"
    obj_connection_id: str = OBJ_CONNECTION_ID,     # e.g., "project.us.ai_conn"
    mm_model_full_id: str = MM_Embedding_MODEL_ID,  # e.g., "project.dataset.mm_embedding_model"
    id_col: str = "product_id",
    image_col: str = "image",        # filename like "foo.jpg"
    uri_col: str = "uri",            # gs:// URL will be stored here
    embedding_col: str = "img_embedding",
):
    """
    Create and backfill image embeddings for products.

    Steps:
      - Ensure `{uri_col}` exists and fill it from `{gcs_prefix}` + `{image_col}` where NULL.
      - Ensure `{embedding_col}` exists.
      - Generate embeddings via ML.GENERATE_EMBEDDING over signed-access URLs built with OBJ.*.

    Args:
      table_full_id / gcs_prefix / obj_connection_id / mm_model_full_id: BigQuery + GCS config.
      id_col / image_col / uri_col / embedding_col: Column names to use.
    """
    gcs_prefix = _normalize_gcs_prefix(gcs_prefix)

    # 1) Add URI column if missing
    sql_add_uri = f"""
    ALTER TABLE `{table_full_id}`
    ADD COLUMN IF NOT EXISTS {uri_col} STRING;
    """
    run_query(sql_add_uri)

    # 2) Fill URI only where it is NULL and image filename is present
    #    Resulting URI looks like: gs://<bucket>/images/<filename>
    if gcs_prefix:
        sql_fill_uri = f"""
        UPDATE `{table_full_id}`
        SET {uri_col} = CONCAT('{gcs_prefix}', {image_col})
        WHERE {uri_col} IS NULL
          AND {image_col} IS NOT NULL
          AND {image_col} != '';
        """
        run_query(sql_fill_uri)

    # 3) Add image embedding column if missing
    sql_add_img_emb = f"""
    ALTER TABLE `{table_full_id}`
    ADD COLUMN IF NOT EXISTS {embedding_col} ARRAY<FLOAT64>;
    """
    run_query(sql_add_img_emb)

    # 4) Backfill image embeddings
    #    We fetch a *read* URL via OBJ.GET_ACCESS_URL(...) and pass it to the multimodal embedding model.
    sql_backfill_img_emb = f"""
    UPDATE `{table_full_id}` AS t
    SET t.{embedding_col} = s.embedding
    FROM (
      SELECT
        {id_col},
        ml_generate_embedding_result AS embedding
      FROM ML.GENERATE_EMBEDDING(
        MODEL `{mm_model_full_id}`,
        (
          SELECT
            {id_col},
            -- Build a GCS object ref from the {uri_col}, then fetch a signed-access URL
            OBJ.GET_ACCESS_URL(
              OBJ.FETCH_METADATA(OBJ.MAKE_REF({uri_col}, '{obj_connection_id}')),
              'r'
            ) AS content
          FROM `{table_full_id}`
          WHERE {uri_col} IS NOT NULL
            AND {uri_col} != ''
        ),
        STRUCT(TRUE AS flatten_json_output)
      )
    ) AS s
    WHERE t.{id_col} = s.{id_col};
    """
    run_query(sql_backfill_img_emb)


def run_products_builder():
    """
    Orchestrate the full products build:
      1) Text embeddings (title + description) → {TEXT_Embedding_MODEL_ID}
      2) Image embeddings from GCS URIs → {MM_Embedding_MODEL_ID}
    Uses config.py constants for IDs and URIs.
    """
    build_text_embeddings(
        table_full_id=PRODUCTS_TABLE_ID,
        text_model_full_id=TEXT_Embedding_MODEL_ID,
    )

    build_image_embeddings(
        table_full_id=PRODUCTS_TABLE_ID,
        gcs_prefix=GCS_Images_URI,
        obj_connection_id=OBJ_CONNECTION_ID,
        mm_model_full_id=MM_Embedding_MODEL_ID,
    )

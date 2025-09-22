from services.bq import run_query
from config import PRODUCTS_TABLE_ID, GCS_Images_URI,OBJ_CONNECTION_ID, TEXT_Embedding_MODEL_ID, MM_Embedding_MODEL_ID

def build_text_embeddings(
    table: str = "products",
    model_name: str = "text_embedding_model",
):
    add_col_sql = f"""
    ALTER TABLE `{table}`
    ADD COLUMN IF NOT EXISTS text_embedding ARRAY<FLOAT64>;

    UPDATE `{table}` AS t
    SET t.text_embedding = s.embedding
    FROM (
      SELECT
        product_id,
        ml_generate_embedding_result AS embedding
      FROM ML.GENERATE_EMBEDDING(
        MODEL `{model_name}`,
        (
          SELECT
            product_id,
            CONCAT(title, ' ', description) AS content
          FROM `{table}`
        ),
        STRUCT(TRUE AS flatten_json_output)
      )
    ) AS s
    WHERE t.product_id = s.product_id;
    """
    run_query(add_col_sql)
  
def build_image_embeddings(
    table_full_id: str,                      # e.g., "project.dataset.products"
    gcs_prefix: str,                       # prefix for images in GCS   e.g.,"g:://images"
    obj_connection_id: str,               # BigQuery remote connection for OBJ.*
    mm_model_full_id: str,                # e.g., "project.dataset.mm_embedding_model"
    id_col: str = "product_id",
    image_col: str = "image",           
    uri_col: str = "uri",                    # will store the gs:// URL
    embedding_col: str = "img_embedding",
    
    
):
    """
    Pipeline:
      1) Add a URI column if it doesn't exist.
      2) Populate URI from the image filename once (only where URI is NULL).
      3) Add an image-embedding column if it doesn't exist.
      4) Backfill embeddings only for rows where the embedding is still NULL.

    Requirements:
      - The table, model, and connection should be in compatible regions (e.g., US).
      - The GCS URIs must be readable by the connection used in OBJ.MAKE_REF.
    """
   

    # 1) Add URI column if missing
    sql_add_uri = f"""
    ALTER TABLE `{table_full_id}`
    ADD COLUMN IF NOT EXISTS {uri_col} STRING;
    """
    run_query(sql_add_uri)

    
    # 2) Fill URI for rows that don't have it yet
    sql_fill_uri = f"""
    UPDATE `{table_full_id}`
    SET {uri_col} = CONCAT('{gcs_prefix}', {image_col})
    WHERE {uri_col} IS  NULL
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

    # 4) Backfill embeddings only where still NULL
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
            OBJ.GET_ACCESS_URL(
              OBJ.FETCH_METADATA(OBJ.MAKE_REF({uri_col}, '{obj_connection_id}')),
              'r'
            ) AS content
          FROM `{table_full_id}`
        ),
        STRUCT(TRUE AS flatten_json_output)
      )
    ) AS s
    WHERE t.{id_col} = s.{id_col}
    """
    run_query(sql_backfill_img_emb)


def run_products_builder():
    build_text_embeddings(PRODUCTS_TABLE_ID,TEXT_Embedding_MODEL_ID)
    build_image_embeddings(table_full_id=PRODUCTS_TABLE_ID,
                           gcs_prefix=GCS_Images_URI,
                           obj_connection_id=OBJ_CONNECTION_ID,
                           mm_model_full_id=MM_Embedding_MODEL_ID)
    
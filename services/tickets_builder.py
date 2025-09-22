from services.bq import run_query
from config import (
    PROJECT_ID,
    DATASET_ID,
    OBJ_CONNECTION_ID,
    GCS_CALLS_URI,    
    TRANSCRIPTION_MODEL_ID,
    BQ_GENERATIVE_MODEL_ID,
    TEXT_Embedding_MODEL_ID    
)
# ========= 1) Create External Object Table (GCS -> BigQuery) =========
def create_call_object_table(
    project_id: str,           
    dataset_id: str,
    connection_id: str,           # e.g. "project.us.audio_conn"
    gcs_call_uri: str,            # e.g. "gs://customers_calls/*"
    table:  str = "call_object_table",
):
    """
    Creates/overwrites an external OBJECT table pointing to audio files in GCS.
    """
    sql = f"""
    CREATE OR REPLACE EXTERNAL TABLE `{project_id}.{dataset_id}.{table}`
    WITH CONNECTION `{connection_id}`
    OPTIONS (
      object_metadata = 'SIMPLE',
      uris = ['{gcs_call_uri}'],
      metadata_cache_mode = 'AUTOMATIC',
      max_staleness = INTERVAL 1 HOUR
    );
    """
    return run_query(sql)


# ========= 2) Transcribe and Save Results =========
def transcribe_calls_to_table(
    project_id: str,           
    dataset_id: str,
    transcribe_model_id: str,        # e.g. "project.dataset.transcribe_model"
    goal_table:  str = "call_transcriptions2",
    source_table: str = "call_object_table2"
    
):
    """
    Runs ML.TRANSCRIBE over the object table and writes results into a standard table.
    """
    sql = f"""
    CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{goal_table}` AS
    SELECT *
    FROM ML.TRANSCRIBE(
      MODEL `{transcribe_model_id}`,
      TABLE `{project_id}.{dataset_id}.{source_table}`
    );
    """
    return run_query(sql)

# ========= 3) Build tickets table ===========
def build_tickets_table(
    project_id: str,           
    dataset_id: str,
    gen_model_id: str,
    call_transcript_column: str = "transcripts",
    main_table:  str = "tickets",
):
    sql = f"""
    CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{main_table}` AS
    WITH
    forms_prepped AS (
      SELECT
        'form' AS source,
        subject,
        body,
        answer,
        type,
        priority
      FROM `{project_id}.{dataset_id}.forms`
    ),
    from_chats AS (
      SELECT
        'chat' AS source,
        subject,
        body,
        answer,
        type,
        priority
      FROM AI.GENERATE_TABLE(
        MODEL `{gen_model_id}`,
        (
          SELECT
            CONCAT(
              'You will receive a raw customer chat transcript between a customer and a support agent.',
              '\\nReturn a compact record with: ',
              'subject (short title), body (1–3 sentences problem), answer (2–5 sentences reply), ',
              'type in {{"Billing and Payments","Technical Support","Account & Access","Shipping & Returns","Product Inquiry","Other"}}, ',
              'priority in {{"high","medium","low"}}. ',
              'Remove PII and ignore greetings/signatures.',
              '\\n\\nCHAT:\\n', chat
            ) AS prompt
          FROM `{project_id}.{dataset_id}.chats`
        ),
        STRUCT(
          'subject STRING, body STRING, answer STRING, type STRING, priority STRING' AS output_schema
        )
      )
    ),
    from_emails AS (
      SELECT
        'email' AS source,
        subject,
        body,
        answer,
        type,
        priority
      FROM AI.GENERATE_TABLE(
        MODEL `{gen_model_id}`,
        (
          SELECT
            CONCAT(
              'You will receive a customer email message.',
              '\\nReturn: subject, body (1–3 sentences), answer (2–5 sentences), ',
              'type in {{"Billing and Payments","Technical Support","Account & Access","Shipping & Returns","Product Inquiry","Other"}}, ',
              'priority in {{"high","medium","low"}}. ',
              'Remove PII and ignore signatures/disclaimers/quoted history.',
              '\\n\\nEMAIL:\\n', email
            ) AS prompt
          FROM `{project_id}.{dataset_id}.emails`
        ),
        STRUCT(
          'subject STRING, body STRING, answer STRING, type STRING, priority STRING' AS output_schema
        )
      )
    ),
    from_calls AS (
      SELECT
        'call' AS source,
        subject,
        body,
        answer,
        type,
        priority
      FROM AI.GENERATE_TABLE(
        MODEL `{gen_model_id}`,
        (
          SELECT
            CONCAT(
              'You will receive a customer call transcript.',
              '\\nSummarize to: subject, body (1–3 sentences), answer (2–5 sentences), ',
              'type in {{"Billing and Payments","Technical Support","Account & Access","Shipping & Returns","Product Inquiry","Other"}}, ',
              'priority in {{"high","medium","low"}}. ',
              'Remove PII.',
              '\\n\\nTRANSCRIPT:\\n', {call_transcript_column}
            ) AS prompt
          FROM `{project_id}.{dataset_id}.call_transcriptions2`
        ),
        STRUCT(
          'subject STRING, body STRING, answer STRING, type STRING, priority STRING' AS output_schema
        )
      )
    ),
    union_all AS (
      SELECT * FROM forms_prepped
      UNION ALL SELECT * FROM from_chats
      UNION ALL SELECT * FROM from_emails
      UNION ALL SELECT * FROM from_calls
    )
    SELECT
      ROW_NUMBER() OVER() AS ticket_id,
      source,
      subject,
      body,
      answer,
      type,
      priority
    FROM union_all
    WHERE
      subject IS NOT NULL AND subject != '' AND
      body IS NOT NULL AND body != '' AND
      answer IS NOT NULL AND answer != '';
    """

    return run_query(sql)

# ========= 4) Build text embeddings =========
def build_text_embeddings(
    project_id: str,           
    dataset_id: str,
    text_embedding_model: str,
    table:  str = "tickets",
    
):
    """
    Adds `text_embedding` (ARRAY<FLOAT64>) to the tickets table if missing,
    then backfills it using ML.GENERATE_EMBEDDING over concatenated text columns.
    """ 

    sql = f"""
    ALTER TABLE `{project_id}.{dataset_id}.{table}`
    ADD COLUMN IF NOT EXISTS text_embedding ARRAY<FLOAT64>;

    UPDATE `{project_id}.{dataset_id}.{table}` AS t
    SET t.text_embedding = s.embedding
    FROM (
      SELECT
        ticket_id,
        ml_generate_embedding_result AS embedding
      FROM ML.GENERATE_EMBEDDING(
        MODEL `{text_embedding_model}`,
        (
          SELECT
            ticket_id,
            CONCAT(subject, ' ', body) AS content
          FROM `{project_id}.{dataset_id}.{table}`
        ),
        STRUCT(TRUE AS flatten_json_output)
      )
    ) AS s
    WHERE t.ticket_id = s.ticket_id;
    """
    run_query(sql)

def run_tickets_builder():
    
    create_call_object_table(
    project_id=PROJECT_ID,
    dataset_id=DATASET_ID,
    connection_id=OBJ_CONNECTION_ID,
    gcs_call_uri=GCS_CALLS_URI,
  )
    
    transcribe_calls_to_table(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        transcribe_model_id=TRANSCRIPTION_MODEL_ID 
    )
    
    
    build_tickets_table(
        project_id=PROJECT_ID,           
        dataset_id=DATASET_ID,
        gen_model_id=BQ_GENERATIVE_MODEL_ID,

    )
    
    build_text_embeddings(
        project_id=PROJECT_ID,           
        dataset_id=DATASET_ID,
        text_embedding_model=TEXT_Embedding_MODEL_ID,
    )
    

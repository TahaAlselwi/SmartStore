# services/tickets_builder.py
"""
Tickets Pipeline (Unification + Embeddings)

Flow:
  1) Create EXTERNAL OBJECT TABLE over call audio in GCS.
  2) Transcribe calls with ML.TRANSCRIBE into a standard table.
  3) Normalize chats/emails/calls with AI.GENERATE_TABLE to a common schema.
  4) UNION with forms (already structured) → write `tickets`.
  5) Add text embeddings over CONCAT(subject, ' ', body).

Notes:
- Dataset, models, and connection should be in the same region (e.g., US).
- The connection used for OBJ.* and Speech must have required IAM roles.
"""

from __future__ import annotations

from services.bq import run_query
from config import (
    PROJECT_ID,
    DATASET_ID,
    OBJ_CONNECTION_ID,
    GCS_CALLS_URI,
    TRANSCRIPTION_MODEL_ID,
    BQ_GENERATIVE_MODEL_ID,
    TEXT_Embedding_MODEL_ID,
)

# ----------------------------- Helpers -----------------------------
def _full_id(project_id: str, dataset_id: str, table: str) -> str:
    return f"{project_id}.{dataset_id}.{table}"


# ---------------------- (1) OBJECT TABLE (calls) ----------------------
def create_call_object_table(
    project_id: str,
    dataset_id: str,
    connection_id: str,              # e.g. "project.us.ai_conn"
    gcs_call_uri: str,               # e.g. "gs://bucket/calls/*"
    table: str = "call_object_table",
):
    """
    Create/replace an external OBJECT table over GCS audio files.
    """
    sql = f"""
    -- External OBJECT table over GCS (audio assets)
    CREATE OR REPLACE EXTERNAL TABLE `{_full_id(project_id, dataset_id, table)}`
    WITH CONNECTION `{connection_id}`
    OPTIONS (
      object_metadata = 'SIMPLE',
      uris = ['{gcs_call_uri}'],
      metadata_cache_mode = 'AUTOMATIC',
      max_staleness = INTERVAL 1 HOUR
    );
    """
    return run_query(sql)


# ---------------------- (2) TRANSCRIBE (calls) ----------------------
def transcribe_calls_to_table(
    project_id: str,
    dataset_id: str,
    transcribe_model_id: str,         # e.g. "project.dataset.transcription_model"
    goal_table: str = "call_transcriptions",
    source_table: str = "call_object_table",
):
    """
    Run ML.TRANSCRIBE over the OBJECT table and write transcripts to a standard table.
    """
    sql = f"""
    -- Transcribe all audio from OBJECT table into goal_table
    CREATE OR REPLACE TABLE `{_full_id(project_id, dataset_id, goal_table)}` AS
    SELECT *
    FROM ML.TRANSCRIBE(
      MODEL `{transcribe_model_id}`,
      TABLE `{_full_id(project_id, dataset_id, source_table)}`
    );
    """
    return run_query(sql)


# ---------------------- (3–4) UNIFIED TICKETS ----------------------
def build_tickets_table(
    project_id: str,
    dataset_id: str,
    gen_model_id: str,                 # e.g. "project.dataset.generative_model"
    call_transcripts_table: str = "call_transcriptions",
    call_transcript_column: str = "transcripts",
    main_table: str = "tickets",
):
    FORMS = _full_id(project_id, dataset_id, "forms")
    CHATS = _full_id(project_id, dataset_id, "chats")
    EMAILS = _full_id(project_id, dataset_id, "emails")
    CALLS = _full_id(project_id, dataset_id, call_transcripts_table)
    TARGET = _full_id(project_id, dataset_id, main_table)

    sql = f"""
    CREATE OR REPLACE TABLE `{TARGET}` AS
    WITH
      -- Forms passthrough
      forms_prepped AS (
        SELECT
          'form' AS source,
          subject,
          body,
          answer,
          type,
          priority
        FROM `{FORMS}`
      ),

      -- Chats → normalize via AI.GENERATE_TABLE
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
            FROM `{CHATS}`
          ),
          STRUCT('subject STRING, body STRING, answer STRING, type STRING, priority STRING' AS output_schema)
        )
      ),

      -- Emails → normalize via AI.GENERATE_TABLE
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
            FROM `{EMAILS}`
          ),
          STRUCT('subject STRING, body STRING, answer STRING, type STRING, priority STRING' AS output_schema)
        )
      ),

      -- Calls → normalize via AI.GENERATE_TABLE
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
            FROM `{CALLS}`
          ),
          STRUCT('subject STRING, body STRING, answer STRING, type STRING, priority STRING' AS output_schema)
        )
      ),

      -- Union of all sources
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
      body    IS NOT NULL AND body    != '' AND
      answer  IS NOT NULL AND answer  != '';
    """
    return run_query(sql)
# ---------------------- (5) TEXT EMBEDDINGS ----------------------
def build_text_embeddings(
    project_id: str,
    dataset_id: str,
    text_embedding_model: str,         # e.g. "project.dataset.text_embedding_model"
    table: str = "tickets",
    embedding_col: str = "text_embedding",
):
    """
    Add `text_embedding ARRAY<FLOAT64>` to tickets if missing,
    then backfill it.
    """
    TARGET = _full_id(project_id, dataset_id, table)

    # Add column if missing
    sql_add = f"""
    ALTER TABLE `{TARGET}`
    ADD COLUMN IF NOT EXISTS {embedding_col} ARRAY<FLOAT64>;
    """
    run_query(sql_add)

    # Backfill embeddings only where NULL
    sql_fill = f"""
    UPDATE `{TARGET}` AS t
    SET t.{embedding_col} = s.embedding
    FROM (
      SELECT
        ticket_id,
        ml_generate_embedding_result AS embedding
      FROM ML.GENERATE_EMBEDDING(
        MODEL `{text_embedding_model}`,
        (
          SELECT
            ticket_id,
            CONCAT(COALESCE(subject, ''), ' ', COALESCE(body, '')) AS content
          FROM `{TARGET}`
        ),
        STRUCT(TRUE AS flatten_json_output)
      )
    ) AS s
    WHERE t.ticket_id = s.ticket_id;
    """
    run_query(sql_fill)


# --------------------------- Orchestrator ---------------------------
def run_tickets_builder():
    """
    Execute the full pipeline with no guards:
      1) OBJECT table over GCS calls.
      2) TRANSCRIBE to `call_transcriptions`.
      3) Build unified `tickets` from forms/chats/emails/calls.
      4) Add text embeddings.
    """
    create_call_object_table(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        connection_id=OBJ_CONNECTION_ID,
        gcs_call_uri=GCS_CALLS_URI,
        table="call_object_table",
    )

    transcribe_calls_to_table(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        transcribe_model_id=TRANSCRIPTION_MODEL_ID,
        goal_table="call_transcriptions",
        source_table="call_object_table",
    )
    
    build_tickets_table(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        gen_model_id=BQ_GENERATIVE_MODEL_ID,
        call_transcripts_table="call_transcriptions",
        call_transcript_column="transcripts",
        main_table="tickets",
    )
    
    build_text_embeddings(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        text_embedding_model=TEXT_Embedding_MODEL_ID,
        table="tickets",
        embedding_col="text_embedding",
    )
    
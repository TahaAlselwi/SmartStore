# services/dashboard_helper.py
"""
Dashboard Helpers
-----------------
A collection of BigQuery SQL helpers that return pandas DataFrames for the
Support Dashboard. All functions delegate execution to `services.bq.run_query_to_df`.

Sections:
  1) Tickets KPIs & Distributions (sentiment, priority, type)
  2) Tickets Churn Risk (via AI.GENERATE_DOUBLE)
  3) Products Content Moderation (abusive language)
  4) Products Satisfaction (boolean via AI.GENERATE_BOOL)
  

Notes:
- `connection_id` and `endpoint` are passed through to BigQuery AI functions.
- Keep dataset, models and connections in the same region (e.g., US).
"""

from __future__ import annotations

import pandas as pd
from typing import Optional
from services.bq import run_query_to_df


# ---------- Shared ----------
def _full_id(project_id: str, dataset_id: str, table: str) -> str:
    """Return fully-qualified table ID."""
    return f"{project_id}.{dataset_id}.{table}"


# ============================================================================
# 1) TICKETS — Sentiment & KPIs
# ============================================================================

def kpis_tickets(
    project_id: str,
    dataset_id: str,
    connection_id: str = "us.my_conn",
    endpoint: str = "gemini-2.5-flash",
) -> pd.DataFrame:
    """
    Compute ticket KPIs:
      - total_tickets
      - negative_rate  (share of 'negative')
      - urgent_count   (priority in {high, urgent, p1, critical})

    Uses AI.GENERATE to classify sentiment as Positive / Neutral / Negative.
    """
    TICKETS = _full_id(project_id, dataset_id, "tickets")
    sql = f"""
    -- Prepare text and normalized priority
    WITH prepared AS (
      SELECT
        ticket_id,
        COALESCE(CONCAT(subject, ' ', body), body, subject, '') AS text_for_sentiment,
        LOWER(COALESCE(priority, 'unknown')) AS priority
      FROM `{TICKETS}`
    ),
    -- Score sentiment via AI.GENERATE (one-word response)
    scored AS (
      SELECT
        ticket_id,
        priority,
        AI.GENERATE(
          CONCAT(
            'Classify the customer sentiment as exactly one of: Positive, Neutral, Negative.',
            ' Respond with one word only.',
            '\\n\\nText: ',
            text_for_sentiment
          ),
          connection_id => '{connection_id}',
          endpoint      => '{endpoint}'
        ).result AS sentiment
      FROM prepared
    ),
    base AS (
      SELECT LOWER(COALESCE(sentiment, 'unknown')) AS sentiment, priority
      FROM scored
    )
    SELECT
      COUNT(*) AS total_tickets,
      SAFE_DIVIDE(COUNTIF(sentiment = 'negative'), COUNT(*)) AS negative_rate,
      COUNTIF(priority IN ('high','urgent','p1','critical')) AS urgent_count
    FROM base
    """
    return run_query_to_df(sql)


def dist_sentiment_tickets(
    project_id: str,
    dataset_id: str,
    connection_id: str = "us.my_conn",
    endpoint: str = "gemini-2.5-flash",
) -> pd.DataFrame:
    """
    Sentiment distribution across tickets:
      - sentiment, cnt

    Uses AI.GENERATE to classify each ticket as Positive / Neutral / Negative.
    """
    TICKETS = _full_id(project_id, dataset_id, "tickets")
    sql = f"""
    WITH prepared AS (
      SELECT
        ticket_id,
        COALESCE(CONCAT(subject, ' ', body), body, subject, '') AS text_for_sentiment
      FROM `{TICKETS}`
    ),
    scored AS (
      SELECT
        ticket_id,
        AI.GENERATE(
          CONCAT(
            'Classify the customer sentiment as exactly one of: Positive, Neutral, Negative.',
            ' Respond with one word only.',
            '\\n\\nText: ',
            text_for_sentiment
          ),
          connection_id => '{connection_id}',
          endpoint      => '{endpoint}'
        ).result AS sentiment
      FROM prepared
    )
    SELECT COALESCE(sentiment, 'unknown') AS sentiment, COUNT(*) AS cnt
    FROM scored
    GROUP BY sentiment
    ORDER BY cnt DESC
    """
    return run_query_to_df(sql)


def priority_distribution_tickets(project_id: str, dataset_id: str) -> pd.DataFrame:
    """
    Priority distribution:
      - priority, cnt, pct
    """
    TICKETS = _full_id(project_id, dataset_id, "tickets")
    sql = f"""
    WITH base AS (
      SELECT COALESCE(priority, 'unknown') AS priority
      FROM `{TICKETS}`
    ),
    agg AS (
      SELECT priority, COUNT(*) AS cnt FROM base GROUP BY priority
    ),
    total AS (
      SELECT SUM(cnt) AS total_cnt FROM agg
    )
    SELECT a.priority, a.cnt, SAFE_DIVIDE(a.cnt, t.total_cnt) AS pct
    FROM agg a CROSS JOIN total t
    ORDER BY a.cnt DESC
    """
    return run_query_to_df(sql)


def type_top5_tickets(project_id: str, dataset_id: str) -> pd.DataFrame:
    """
    Top 5 ticket types:
      - type, cnt
    """
    TICKETS = _full_id(project_id, dataset_id, "tickets")
    sql = f"""
    SELECT COALESCE(type, 'unknown') AS type, COUNT(*) AS cnt
    FROM `{TICKETS}`
    GROUP BY type
    ORDER BY cnt DESC
    LIMIT 5
    """
    return run_query_to_df(sql)

# ============================================================================
# 2) TICKETS — Churn Risk (AI.GENERATE_DOUBLE in [0.0, 1.0])
# ============================================================================

def churn_kpis_tickets(
    project_id: str,
    dataset_id: str,
    high_threshold: float = 0.7,
    connection_id: str = "us.my_conn",
    endpoint: str = "gemini-2.5-flash",
) -> pd.DataFrame:
    """
    Churn KPIs across tickets:
      - avg_risk
      - p90_risk
      - high_risk_count (>= threshold)
      - total_tickets
    """
    TICKETS = _full_id(project_id, dataset_id, "tickets")
    sql = f"""
    WITH prepared AS (
      SELECT
        ticket_id,
        COALESCE(CONCAT(subject, ' ', body), body, subject, '') AS text
      FROM `{TICKETS}`
    ),
    scored AS (
      SELECT
        ticket_id,
        AI.GENERATE_DOUBLE(
          CONCAT(
            'Return a probability between 0.0 and 1.0 of customer churn. ',
            'Output number only. Text: ',
            text
          ),
          connection_id => '{connection_id}',
          endpoint      => '{endpoint}'
        ).result AS raw_risk
      FROM prepared
    ),
    clean AS (
      -- Clamp to [0,1] to guard against any model drift
      SELECT ticket_id, LEAST(1.0, GREATEST(0.0, raw_risk)) AS risk FROM scored
    )
    SELECT
      AVG(risk) AS avg_risk,
      APPROX_QUANTILES(risk, 100)[OFFSET(90)] AS p90_risk,
      SUM(CASE WHEN risk >= {high_threshold} THEN 1 ELSE 0 END) AS high_risk_count,
      COUNT(*) AS total_tickets
    FROM clean
    """
    return run_query_to_df(sql)


def churn_distribution_tickets(
    project_id: str,
    dataset_id: str,
    bucket_size: float = 0.2,  # kept for API compatibility; buckets are fixed below
    connection_id: str = "us.my_conn",
    endpoint: str = "gemini-2.5-flash",
) -> pd.DataFrame:
    """
    Churn distribution into five fixed buckets:
      - [0.0–0.2), [0.2–0.4), [0.4–0.6), [0.6–0.8), [0.8–1.0]
    """
    TICKETS = _full_id(project_id, dataset_id, "tickets")
    sql = f"""
    WITH prepared AS (
      SELECT
        ticket_id,
        COALESCE(CONCAT(subject, ' ', body), body, subject, '') AS text
      FROM `{TICKETS}`
    ),
    scored AS (
      SELECT
        ticket_id,
        AI.GENERATE_DOUBLE(
          CONCAT(
            'Return a probability between 0.0 and 1.0 of customer churn. ',
            'Output number only. Text: ',
            text
          ),
          connection_id => '{connection_id}',
          endpoint      => '{endpoint}'
        ).result AS raw_risk
      FROM prepared
    ),
    clean AS (
      SELECT ticket_id, LEAST(1.0, GREATEST(0.0, raw_risk)) AS risk FROM scored
    ),
    bucketed AS (
      SELECT
        CASE
          WHEN risk < 0.2 THEN '[0.0–0.2)'
          WHEN risk < 0.4 THEN '[0.2–0.4)'
          WHEN risk < 0.6 THEN '[0.4–0.6)'
          WHEN risk < 0.8 THEN '[0.6–0.8)'
          ELSE '[0.8–1.0]'
        END AS bucket_label
      FROM clean
    )
    SELECT bucket_label, COUNT(*) AS cnt
    FROM bucketed
    GROUP BY bucket_label
    ORDER BY bucket_label
    """
    return run_query_to_df(sql)


def churn_top_tickets(
    project_id: str,
    dataset_id: str,
    limit: int = 20,
    connection_id: str = "us.my_conn",
    endpoint: str = "gemini-2.5-flash",
) -> pd.DataFrame:
    """
    Top-N risky tickets:
      - ticket_id, risk (0–1), snippet (first 200 chars)
    """
    TICKETS = _full_id(project_id, dataset_id, "tickets")
    sql = f"""
    WITH prepared AS (
      SELECT
        ticket_id,
        COALESCE(CONCAT(subject, ' ', body), body, subject, '') AS text
      FROM `{TICKETS}`
    ),
    scored AS (
      SELECT
        ticket_id,
        AI.GENERATE_DOUBLE(
          CONCAT(
            'Return a probability between 0.0 and 1.0 of customer churn. ',
            'Output number only. Text: ',
            text
          ),
          connection_id => '{connection_id}',
          endpoint      => '{endpoint}'
        ).result AS raw_risk,
        text
      FROM prepared
    ),
    clean AS (
      SELECT
        ticket_id,
        LEAST(1.0, GREATEST(0.0, raw_risk)) AS risk,
        text
      FROM scored
    )
    SELECT
      ticket_id,
      risk,
      SAFE.SUBSTR(text, 1, 200) AS snippet
    FROM clean
    ORDER BY risk DESC
    LIMIT {int(limit)}
    """
    return run_query_to_df(sql)

# ============================================================================
# 3) PRODUCTS — Content Moderation (abusive/profane language)
# ============================================================================

def kpis_abuse_products(
    project_id: str,
    dataset_id: str,
    connection_id: str = "us.my_conn",
    endpoint: str = "gemini-2.5-flash",
) -> pd.DataFrame:
    """
    KPIs on abusive content in product comments:
      - abusive_count
      - total_with_comments
      - abuse_rate
    """
    PRODUCTS = _full_id(project_id, dataset_id, "products")
    sql = f"""
    -- Extract non-empty comments per product
    WITH prepared AS (
      SELECT
        product_id,
        ARRAY(
          SELECT TRIM(c)
          FROM UNNEST(customer_comments) c
          WHERE c IS NOT NULL AND LENGTH(TRIM(c)) > 0
        ) AS comments_arr
      FROM `{PRODUCTS}`
    ),
    joined AS (
      SELECT
        product_id,
        ARRAY_TO_STRING(comments_arr, '\\n') AS comments_text
      FROM prepared
      WHERE ARRAY_LENGTH(comments_arr) > 0
    ),
    -- AI moderation (boolean)
    moderation AS (
      SELECT
        product_id,
        AI.GENERATE_BOOL(
          CONCAT(
            'Does this text contain abusive/profane language? true/false only. Text: ',
            comments_text
          ),
          connection_id => '{connection_id}',
          endpoint      => '{endpoint}'
        ).result AS abusive
      FROM joined
    )
    SELECT
      SUM(CASE WHEN abusive THEN 1 ELSE 0 END) AS abusive_count,
      COUNT(*) AS total_with_comments,
      SAFE_DIVIDE(SUM(CASE WHEN abusive THEN 1 ELSE 0 END), NULLIF(COUNT(*),0)) AS abuse_rate
    FROM moderation
    """
    return run_query_to_df(sql)


def abuse_distribution_products(
    project_id: str,
    dataset_id: str,
    connection_id: str = "us.my_conn",
    endpoint: str = "gemini-2.5-flash",
) -> pd.DataFrame:
    """
    Distribution of abusive vs non-abusive products:
      - abusive (BOOL), cnt
    """
    PRODUCTS = _full_id(project_id, dataset_id, "products")
    sql = f"""
    WITH prepared AS (
      SELECT
        product_id,
        ARRAY(
          SELECT TRIM(c)
          FROM UNNEST(customer_comments) c
          WHERE c IS NOT NULL AND LENGTH(TRIM(c)) > 0
        ) AS comments_arr
      FROM `{PRODUCTS}`
    ),
    joined AS (
      SELECT
        product_id,
        ARRAY_TO_STRING(comments_arr, '\\n') AS comments_text
      FROM prepared
      WHERE ARRAY_LENGTH(comments_arr) > 0
    ),
    moderation AS (
      SELECT
        product_id,
        AI.GENERATE_BOOL(
          CONCAT(
            'Does this text contain abusive/profane language? true/false only. Text: ',
            comments_text
          ),
          connection_id => '{connection_id}',
          endpoint      => '{endpoint}'
        ).result AS abusive
      FROM joined
    )
    SELECT abusive, COUNT(*) AS cnt
    FROM moderation
    GROUP BY abusive
    ORDER BY cnt DESC
    """
    return run_query_to_df(sql)


def abusive_examples_products(
    project_id: str,
    dataset_id: str,
    limit: int = 20,
    connection_id: str = "us.my_conn",
    endpoint: str = "gemini-2.5-flash",
) -> pd.DataFrame:
    """
    Examples of products flagged as abusive:
      - product_id, comment_snippet (first 300 chars), abusive (TRUE)
    """
    PRODUCTS = _full_id(project_id, dataset_id, "products")
    sql = f"""
    WITH prepared AS (
      SELECT
        product_id,
        ARRAY(
          SELECT TRIM(c)
          FROM UNNEST(customer_comments) c
          WHERE c IS NOT NULL AND LENGTH(TRIM(c)) > 0
        ) AS comments_arr
      FROM `{PRODUCTS}`
    ),
    joined AS (
      SELECT
        product_id,
        ARRAY_TO_STRING(comments_arr, ' | ') AS comments_text
      FROM prepared
      WHERE ARRAY_LENGTH(comments_arr) > 0
    ),
    moderation AS (
      SELECT
        product_id,
        comments_text,
        AI.GENERATE_BOOL(
          CONCAT(
            'Does this text contain abusive/profane language? true/false only. Text: ',
            comments_text
          ),
          connection_id => '{connection_id}',
          endpoint      => '{endpoint}'
        ).result AS abusive
      FROM joined
    )
    SELECT
      product_id,
      SAFE.SUBSTR(comments_text, 1, 300) AS comment_snippet,
      abusive
    FROM moderation
    WHERE abusive = TRUE
    ORDER BY product_id
    LIMIT {int(limit)}
    """
    return run_query_to_df(sql)


# ============================================================================
# 4) PRODUCTS — Satisfaction (boolean via AI.GENERATE_BOOL)
# ============================================================================

def satisfaction_bool_kpis_products(
    project_id: str,
    dataset_id: str,
    connection_id: str = "us.my_conn",
    endpoint: str = "gemini-2.5-flash",
) -> pd.DataFrame:
    """
    Overall satisfaction KPIs (boolean interpretation):
      - satisfied_count
      - total_scored
      - satisfaction_rate
    """
    PRODUCTS = _full_id(project_id, dataset_id, "products")
    sql = f"""
    WITH prepared AS (
      SELECT
        product_id,
        ARRAY(
          SELECT TRIM(c)
          FROM UNNEST(customer_comments) c
          WHERE c IS NOT NULL AND LENGTH(TRIM(c)) > 0
        ) AS comments_arr
      FROM `{PRODUCTS}`
    ),
    joined AS (
      SELECT
        product_id,
        ARRAY_TO_STRING(comments_arr, '\\n') AS comments_text
      FROM prepared
      WHERE ARRAY_LENGTH(comments_arr) > 0
    ),
    scored AS (
      SELECT
        product_id,
        AI.GENERATE_BOOL(
          CONCAT(
            'Is the overall customer satisfaction with this product positive? Answer strictly true or false. ',
            'Consider all comments together. Text: ',
            comments_text
          ),
          connection_id => '{connection_id}',
          endpoint      => '{endpoint}'
        ).result AS satisfied
      FROM joined
    ),
    agg AS (
      SELECT
        SUM(CASE WHEN satisfied THEN 1 ELSE 0 END) AS satisfied_count,
        COUNT(*) AS total_scored
      FROM scored
    )
    SELECT
      satisfied_count,
      total_scored,
      SAFE_DIVIDE(satisfied_count, NULLIF(total_scored, 0)) AS satisfaction_rate
    FROM agg
    """
    return run_query_to_df(sql)


def satisfaction_bool_distribution_products(
    project_id: str,
    dataset_id: str,
    connection_id: str = "us.my_conn",
    endpoint: str = "gemini-2.5-flash",
) -> pd.DataFrame:
    """
    Satisfaction distribution (boolean):
      - satisfied (BOOL), cnt
    """
    PRODUCTS = _full_id(project_id, dataset_id, "products")
    sql = f"""
    WITH prepared AS (
      SELECT
        product_id,
        ARRAY(
          SELECT TRIM(c)
          FROM UNNEST(customer_comments) c
          WHERE c IS NOT NULL AND LENGTH(TRIM(c)) > 0
        ) AS comments_arr
      FROM `{PRODUCTS}`
    ),
    joined AS (
      SELECT
        product_id,
        ARRAY_TO_STRING(comments_arr, '\\n') AS comments_text
      FROM prepared
      WHERE ARRAY_LENGTH(comments_arr) > 0
    ),
    scored AS (
      SELECT
        product_id,
        AI.GENERATE_BOOL(
          CONCAT(
            'Is the overall customer satisfaction with this product positive? Answer strictly true or false. ',
            'Consider all comments together. Text: ',
            comments_text
          ),
          connection_id => '{connection_id}',
          endpoint      => '{endpoint}'
        ).result AS satisfied
      FROM joined
    )
    SELECT satisfied, COUNT(*) AS cnt
    FROM scored
    GROUP BY satisfied
    ORDER BY cnt DESC
    """
    return run_query_to_df(sql)




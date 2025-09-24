# pages/1_Agent_Assistant.py
"""
Agent Assistant (Support Side)
- Searches similar historical tickets using BigQuery VECTOR_SEARCH.
- Drafts a concise reply grounded in prior tickets via Vertex AI.

Inputs:
  • Issue text (user description)
  • Top-K similar cases
  • Channel (optional: filter not applied yet; reserved for future)

Outputs:
  • Table of similar tickets
  • Drafted reply
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from services.bq import run_query_to_df
from services.vertex import draft_reply
from config import TICKETS_TABLE_ID, TEXT_Embedding_MODEL_ID

# ---------------------------- Page config ----------------------------
st.set_page_config(page_title="Agent Assistant", page_icon="🎧", layout="wide")
st.title("🎧 Agent Assistant")
st.caption("Find similar cases with BigQuery Vector Search, then draft a grounded reply.")

# ---------------------------- Helpers ----------------------------
def _escape_str(s: str) -> str:
    """Escape a Python string for safe inlining into a BigQuery SQL literal."""
    if s is None:
        return ""
    return str(s).replace("\\", "\\\\").replace('"', '\\"')

# ---------------------------- Form ----------------------------
with st.form("assistant_form"):
    st.markdown("**Describe the issue**")
    issue_text = st.text_area(
        "Issue text",
        value="",
        height=160,
        key="issue_text",
        help="Describe the customer's problem in your own words.",
        placeholder="e.g., My order has been delayed. Can you update me on the expected delivery time?"
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        top_k = st.number_input(
            "Similar cases",
            min_value=1,
            max_value=10,
            value=3,
            key="topk",
            help="Number of most similar past tickets to retrieve."
        )
    with c2:
        channel = st.selectbox(
            "Channel",
            ["Auto-detect", "form", "chat", "email", "call"],
            index=0,
            key="chan",
            help="Reserved for future filtering by ticket source."
        )

    c3, c4 = st.columns([1, 1])
    with c3:
        submitted = st.form_submit_button(
            "Search similar",
            help="Find past tickets with the closest meaning to this issue."
        )
    with c4:
        draft_submitted = st.form_submit_button(
            "Draft reply",
            help="Generate a concise reply grounded in the retrieved tickets."
        )

st.markdown("### Results")

# ---------------------------- Search ----------------------------
if submitted:
    if not issue_text.strip():
        st.warning("Please enter a description of the issue first.")
    else:
        safe_issue = _escape_str(issue_text)
        # Search similar tickets via VECTOR_SEARCH over text embeddings
        sql = f"""
        -- Retrieve Top-K most similar tickets to the provided issue text.
        SELECT base.ticket_id, base.subject, base.body, base.answer,
               base.type, base.priority, base.source, distance
        FROM VECTOR_SEARCH(
          TABLE `{TICKETS_TABLE_ID}`,
          'text_embedding',
          (
            SELECT
              ml_generate_embedding_result,
              content AS query
            FROM ML.GENERATE_EMBEDDING(
              MODEL `{TEXT_Embedding_MODEL_ID}`,
              (SELECT "{safe_issue}" AS content)
            )
          ),
          top_k => {int(top_k)}
        )
        ORDER BY distance ASC
        """
        try:
            df = run_query_to_df(sql)
            st.session_state["search_df"] = df
            if df.empty:
                st.warning("No similar cases found.")
            else:
                cols = ["ticket_id", "subject", "body", "answer", "type", "priority", "source", "distance"]
                cols = [c for c in cols if c in df.columns]
                st.dataframe(df[cols], use_container_width=True, hide_index=True)
        except Exception as e:
            st.error("BigQuery query failed. Please check your config and dataset.")
            st.caption(str(e))

# ---------------------------- Draft reply ----------------------------
if draft_submitted:
    df = st.session_state.get("search_df")
    if df is None or df.empty:
        st.warning("Run the search first to load similar tickets.")
    else:
        # Only the fields needed to ground the reply
        prior = df[["subject", "body", "answer"]].fillna("").to_dict(orient="records")
        prompt = f"""
        You are a senior customer support agent. Draft a concise, empathetic reply using ONLY the prior tickets.

        # User issue
        {issue_text}

        # Prior tickets
        {prior}

        # Instructions
        - Detect the user's language from the issue and reply in the same language.
        - Ground your reply in the 'answer' fields; use 'subject' and 'body' for context.
        - If answers conflict, choose the safest & most general steps.

        # Output
        Return ONLY the final reply text, no headers/JSON/markdown.
        """.strip()
        try:
            reply_text = draft_reply(model_name="gemini-2.5-pro", prompt=prompt)
            if not reply_text:
                st.info("No draft generated. Please try refining the issue description or rerun the search.")
            with st.container(border=True):
                st.subheader("Suggested reply")
                st.text_area("Draft", value=reply_text, height=220, key="static_draft")
        except Exception as e:
            st.error("Vertex call failed. Please verify Vertex init and permissions.")
            st.caption(str(e))

st.markdown("---")
st.caption("© 2025 Smart Store • Built with Streamlit and BigQuery AI")

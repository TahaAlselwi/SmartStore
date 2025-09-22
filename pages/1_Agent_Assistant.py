import streamlit as st
import pandas as pd
from services.bq import run_query_to_df
from services.vertex import draft_reply
from config import TICKETS_TABLE_ID, TEXT_Embedding_MODEL_ID

st.title("🎧 Agent Assistant")
st.caption("Search similar historical cases using BigQuery Vector Search.")

with st.form("assistant_form"):
    st.markdown("**Describe the issue**")
    issue_text = st.text_area("e.g., My order has been delayed. Could you please provide me with an update on the expected delivery time? ", height=160, key="issue_text")

    c1, c2 = st.columns([1, 1])
    with c1:
        top_k = st.number_input("Similar cases", 1, 10, 3, key="topk")
    with c2:
        channel = st.selectbox("Channel", ["Auto-detect", "form", "chat", "email", "call"], index=1, key="chan")

    c3, c4 = st.columns([1, 1])
    with c3:
        submitted = st.form_submit_button("Search similar")
    with c4:
        draft_submitted = st.form_submit_button("Draft reply")

st.markdown("### Results")

if submitted:
    if not issue_text.strip():
        st.warning("Please enter a description of the issue first.")
    else:
        sql = f"""
        SELECT base.ticket_id, base.subject, base.body, base.answer,
          base.type, base.priority, base.source, distance
        FROM VECTOR_SEARCH(
          TABLE `{TICKETS_TABLE_ID}`,
          'text_embedding',
          (
            SELECT ml_generate_embedding_result, content AS query
            FROM ML.GENERATE_EMBEDDING(
              MODEL `{TEXT_Embedding_MODEL_ID}`,
              (SELECT "{issue_text}" AS content)
            )
          ),
          top_k => {top_k}
        )
        ORDER BY distance ASC
        """
        try:
            df = run_query_to_df(sql)
            st.session_state["search_df"] = df
            if df.empty:
                st.warning("No similar cases found.")
            else:
                cols = ["ticket_id", "subject", "body", "answer", "type", "priority","source"]
                st.dataframe(df[[c for c in cols if c in df.columns]], use_container_width=True)
        except Exception as e:
            st.error("BigQuery query failed.")
            st.caption(str(e))

if draft_submitted:
    df = st.session_state.get("search_df")
    if df is None or df.empty:
        st.warning("Run the search first to load similar tickets.")
    else:
        prior = df[["subject", "body", "answer"]].to_dict(orient="records")
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
            with st.container(border=True):
                st.subheader("Suggested reply")
                st.text_area("Draft", value=reply_text, height=220, key="static_draft")
        except Exception as e:
            st.error("Vertex call failed.")
            st.caption(str(e))

st.markdown("---")
st.caption("© 2025 Smart Store • Built with Streamlit and BigQuery AI")

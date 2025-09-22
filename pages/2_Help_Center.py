import streamlit as st
import pandas as pd
from services.bq import run_query_to_df
from services.vertex import draft_reply
from config import TICKETS_TABLE_ID, TEXT_Embedding_MODEL_ID

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Help Center", page_icon="🤖", layout="wide")
st.title("🛟 Help Center")
st.caption("Do you have a question? Type what you’re looking for, and the intelligent system will provide you with the answer or the appropriate support.")

# ----------------------------
# Internal parameters
# ----------------------------
TOP_K = 5
SUPPORT_EMAIL = "support@example.com"
SUPPORT_PHONE = "+90 555 000 0000"


# ----------------------------
# Search helper
# ----------------------------
def find_similar_tickets(issue_text: str, k: int) -> pd.DataFrame:
    sql = f"""
    SELECT
      base.subject, base.body, base.answer, distance
    FROM VECTOR_SEARCH(
      TABLE `{TICKETS_TABLE_ID}`,
      'text_embedding',
      (
        SELECT
          ml_generate_embedding_result,
          content AS query
        FROM ML.GENERATE_EMBEDDING(
          MODEL `{TEXT_Embedding_MODEL_ID}`,
          (SELECT "{issue_text}" AS content)
        )
      ),
      top_k => {k}
    )
    ORDER BY distance ASC
    """
    return run_query_to_df(sql)


# ----------------------------
# UI
# ----------------------------
user_issue = st.text_area(
    "Your question:",
    height=140,
    placeholder="e.g., How can I change my password?"
)

if st.button("Get answer"):
    # Try to fetch similar tickets, but don’t show errors to the customer
    try:
        df = find_similar_tickets(user_issue, TOP_K)
    except Exception:
        df = None

    if df is None or df.empty:
        similar_tickets = []
    else:
        similar_tickets = df.fillna("").to_dict(orient="records")

    # ----------------------------
    # Prompt
    # ----------------------------
    prompt = f"""
                You are a senior customer support agent for an end-user Help Center. You will receive:
                1) The customer's issue.
                2) A small set of similar past tickets.

                Decision policy:
                A) Provide an answer IF:
                - The similar tickets clearly contain a correct, safe, and applicable solution; OR
                - The question is a common, generic FAQ that can be answered with general steps that do not require account/order lookup.
                    Examples:
                    • How to change/reset password
                    • How to update email/phone
                    • How to cancel an order (general policy steps)
                    • Return/refund policy overview
                    • Shipping times (general info)
                When answering, give concise, friendly, step-by-step instructions. Ground in similar tickets when present; otherwise provide general, safe guidance.

                B) Route to support IF:
                - The request is account- or order-specific, time-sensitive, or requires internal lookup/verification
                    (e.g., “The product I ordered two days ago hasn’t arrived,” “Where is my order?”, “Why was my refund not processed?”).
                In these cases, politely instruct the customer to contact support and include:

                Support contact (use exactly this when routing):
                - Email: {SUPPORT_EMAIL}
                - Phone: {SUPPORT_PHONE}

                Language:
                - Detect the customer’s language from the issue and reply in that language.

                # Customer issue
                {user_issue}

                # Similar tickets (array; may be empty)
                {similar_tickets}

                # Output requirements
                Return ONLY one of the following (no headers/JSON/markdown):
                - A concise, friendly solution for the customer (grounded in similar tickets if available, or general FAQ guidance when appropriate).
                - OR a short routing message asking the customer to contact support, including the email and phone above.
                """.strip()

    reply_text = draft_reply(model_name="gemini-2.5-pro", prompt=prompt)
    if not reply_text:
        reply_text = f"Please contact our support team for assistance:\nEmail: {SUPPORT_EMAIL}\nPhone: {SUPPORT_PHONE}"

    with st.container(border=True):
        st.subheader("Answer")
        st.write(reply_text)

st.markdown("---")
st.caption("© 2025 Smart Store • Built with Streamlit and BigQuery AI")
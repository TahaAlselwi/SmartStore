# Home.py
"""
Smart Store — Main Page (Streamlit Multi-Page App)

- Lets users search products by meaning using:
  1) Text query → compare against products.text_embedding via BigQuery VECTOR_SEARCH.
  2) Image upload → compute local image embedding (Vertex) and compare against products.img_embedding.

- Provides admin buttons to (re)build:
  • Product embeddings (text + image)
  • Unified tickets table (+ text embeddings)

Notes:
- CONFIG REQUIRED: set PROJECT_ID, DATASET_ID, PRODUCTS_TABLE_ID, TEXT_Embedding_MODEL_ID in config.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from services.bq import run_query_to_df
from services.vertex import get_image_embedding
from services.products_builder import run_products_builder
from services.tickets_builder import run_tickets_builder
from config import PRODUCTS_TABLE_ID, TEXT_Embedding_MODEL_ID

# ============================== UI Constants ==============================
PAGE_TITLE = "Smart Store"
PAGE_ICON = "🛍️"
CARDS_PER_ROW = 3
DESC_PREVIEW_CHARS = 200

# ============================== Streamlit Setup ===========================
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
st.markdown(
    """
    <div style="padding:8px 0 0 0">
      <h1 style="margin-bottom:0">🛍️ Smart Store</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================== Helpers ==================================
def _escape_str(s: str) -> str:
    """Escape a Python string for safe inlining into a BigQuery SQL literal."""
    if s is None:
        return ""
    # Escape backslashes first, then quotes
    s = str(s).replace("\\", "\\\\").replace('"', '\\"')
    return s

def _format_array(arr: list[float] | np.ndarray) -> str:
    """Format a numeric vector into a BigQuery array literal: [0.1, 0.2, ...]."""
    if arr is None:
        return "[]"
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    return "[" + ", ".join(f"{float(x):.8f}" for x in arr) + "]"

def _safe_text(val) -> str:
    """Normalize text for card descriptions (handles NaN, lists, dicts gracefully)."""
    if val is None:
        return ""
    if isinstance(val, (list, tuple, dict, np.ndarray)):
        return str(val)
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    return str(val)

def _search_by_image(img_bytes: bytes, top_k: int) -> pd.DataFrame | None:
    """
    Compute local image embedding, then run a VECTOR_SEARCH against products.img_embedding.
    Returns a DataFrame or None on failure.
    """
    try:
        img_emb = get_image_embedding(img_bytes)
    except Exception as e:
        st.error(f"Failed to compute image embedding: {e}")
        return None

    if not img_emb:
        st.warning("Could not compute image embedding. Please try another image.")
        return None

    emb_literal = _format_array(img_emb)
    sql = f"""
      SELECT base.product_id, base.title, base.categories, base.description, base.uri, distance
      FROM VECTOR_SEARCH(
        TABLE `{PRODUCTS_TABLE_ID}`,
        'img_embedding',
        (SELECT {emb_literal} AS img_embedding),
        top_k => {int(top_k)}
      )
      ORDER BY distance ASC
    """
    try:
        return run_query_to_df(sql)
    except Exception as e:
        st.error(f"Image search failed: {e}")
        return None

def _search_by_text(query_text: str, top_k: int) -> pd.DataFrame | None:
    """
    Generate text embedding in-query (ML.GENERATE_EMBEDDING) and search products.text_embedding.
    Returns a DataFrame or None on failure.
    """
    safe_q = _escape_str(query_text)
    sql = f"""
      SELECT base.product_id, base.title, base.categories, base.description, base.uri, distance
      FROM VECTOR_SEARCH(
        TABLE `{PRODUCTS_TABLE_ID}`,
        'text_embedding',
        (
          SELECT
            ml_generate_embedding_result,
            content AS query
          FROM ML.GENERATE_EMBEDDING(
            MODEL `{TEXT_Embedding_MODEL_ID}`,
            (SELECT "{safe_q}" AS content)
          )
        ),
        top_k => {int(top_k)}
      )
      ORDER BY distance ASC
    """
    try:
        return run_query_to_df(sql)
    except Exception as e:
        st.error(f"Text search failed: {e}")
        return None

def _render_cards(results_df: pd.DataFrame) -> None:
    """Render product result cards in a responsive grid."""
    rows = int(np.ceil(len(results_df) / CARDS_PER_ROW))
    cards = results_df.to_dict(orient="records")

    for r in range(rows):
        cols = st.columns(CARDS_PER_ROW)
        for i in range(CARDS_PER_ROW):
            idx = r * CARDS_PER_ROW + i
            if idx >= len(cards):
                continue
            item = cards[idx]
            with cols[i]:
                with st.container(border=True):
                    uri = item.get("uri")
                    if isinstance(uri, str) and uri.startswith("gs://"):
                        uri = uri.replace("gs://", "https://storage.googleapis.com/", 1)

                    if isinstance(uri, str) and uri.startswith("http"):
                        st.image(uri, use_container_width=True)

                    st.markdown(f"**{item.get('title','')}**")

                    desc = _safe_text(item.get("description", ""))
                    if desc:
                        preview = desc[:DESC_PREVIEW_CHARS]
                        if len(desc) > DESC_PREVIEW_CHARS:
                            preview += "…"
                        st.write(preview)

                    uid = f"{item.get('product_id','')}_{idx}"
                    st.button("Add to shortlist", key=f"add_{uid}")
                    st.button("Product details", key=f"details_{uid}")

# ============================== Sidebar (Admin) ===========================
with st.sidebar:
    st.markdown("### Control Panel")
    top_k = st.number_input("Similar products", min_value=1, max_value=60, value=9, step=1)
    st.divider()
    st.markdown("### Admin / Builders")

    if st.button("Products Table", key="btn_build_products", help="Rebuild product embeddings (text + image) in BigQuery"):
        with st.spinner("Building product embeddings (text & image)..."):
            try:
                run_products_builder()
                st.success("Product embeddings built/refreshed successfully.")
                st.toast("Product embeddings updated ✅")
            except Exception as e:
                st.error(f"Product embeddings build failed: {e}")

    if st.button("Tickets Table", key="btn_build_tickets", help="Rebuild unified tickets table from chats, emails, forms, calls with embeddings"):
        with st.spinner("Building tickets table..."):
            try:
                run_tickets_builder()
                st.success("Tickets table built/refreshed successfully.")
                st.toast("Tickets table updated ✅")
            except Exception as e:
                st.error(f"Tickets build failed: {e}")

# ============================== Inputs ===================================
query_text = st.text_input(
    "Describe what you're looking for",
    placeholder="e.g., red leather handbag",
)

uploaded_img = st.file_uploader(
    "Or upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
    help="We’ll compute the image embedding locally via Vertex AI (no GCS upload).",
)
if uploaded_img is not None:
    st.image(uploaded_img, caption="Query image", width=280)

go = st.button("Find", type="primary")

# ============================== Execute Search ===========================
results_df: pd.DataFrame | None = None

if go:
    if uploaded_img is not None:
        with st.spinner("Searching by image…"):
            results_df = _search_by_image(uploaded_img.getvalue(), int(top_k))
    elif query_text.strip():
        with st.spinner("Searching by text…"):
            results_df = _search_by_text(query_text, int(top_k))
    else:
        st.warning("Please enter a text query or upload an image.")

# ============================== Display Results ==========================
if results_df is None:
    st.markdown("### Ready to search")
else:
    if results_df.empty:
        st.info("No results found. Try a different query or image.")
    else:
        display_query = "uploaded image" if uploaded_img is not None else query_text
        st.markdown(f"### Results for: _{display_query}_")
        _render_cards(results_df)

st.divider()
st.caption("© 2025 Smart Store • Built with Streamlit and BigQuery AI")

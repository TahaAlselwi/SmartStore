import streamlit as st
import pandas as pd
import numpy as np
from services.bq import run_query_to_df
from services.vertex import get_image_embedding
from services.products_builder import run_products_builder
from services.tickets_builder import run_tickets_builder
from config import PRODUCTS_TABLE_ID, TEXT_Embedding_MODEL_ID, MM_Embedding_MODEL_ID
  
# ---------- Header ----------
st.set_page_config(page_title="Smart Store", page_icon="🛍️", layout="wide")
st.markdown(
    """
    <div style="padding:8px 0 0 0">
      <h1 style="margin-bottom:0">🛍️ Smart Store</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
# ---------- Control Panel (Sidebar) ----------
with st.sidebar:
    st.markdown("### Control Panel")
    top_k = st.number_input("Similar products", min_value=1, max_value=60, value=9, step=1)
    st.divider()
    
    st.markdown("### Admin / Builders")

    if st.button("Products Table", key="btn_build_products"):
        with st.spinner("Building product embeddings (text & image)..."):
            try:
                run_products_builder()
                st.success("Product embeddings built/refreshed successfully.")
                st.toast("Product embeddings updated ✅")
            except Exception as e:
                st.error(f"Product embeddings build failed: {e}")
                
    if st.button("Tickets Table", key="btn_build_tickets"):
        with st.spinner("Building tickets table..."):
            try:
                run_tickets_builder()
                st.success("Tickets table built/refreshed successfully.")
                st.toast("Tickets table updated ✅")
            except Exception as e:
                st.error(f"Tickets build failed: {e}")

# ---------- Inputs ----------
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

# ---------- Vector Search Builder ----------
results_df = None
if go:
    if uploaded_img is not None:
        embedding_col = "img_embedding"
        with st.spinner("Computing image embedding..."):
            try:
                img_bytes = uploaded_img.getvalue()
                img_embedding = get_image_embedding(img_bytes)
            except Exception as e:
                img_embedding = None
                st.error(f"Failed to compute image embedding: {e}")

        if img_embedding:
            query_img = f"""
                SELECT base.product_id, base.title, base.categories, base.description, base.uri, distance
                FROM VECTOR_SEARCH(
                  TABLE `{PRODUCTS_TABLE_ID}`,
                  '{embedding_col}',
                  (SELECT @emb AS {embedding_col}),
                  top_k => {int(top_k)}
                )
                ORDER BY distance ASC
            """
            try:
                results_df = run_query_to_df(
                    {
                        "sql": query_img,
                        "params": [
                            {"name": "emb", "type": "ARRAY<FLOAT64>", "value": img_embedding}
                        ],
                    }
                )
            except Exception:
                # Fallback: inline ARRAY literal if param arrays aren't supported
                try:
                    arr = ", ".join(f"{x:.8f}" for x in img_embedding)
                    query_inline = query_img.replace(
                        "(SELECT @emb AS", f"(SELECT [{arr}] AS"
                    )
                    results_df = run_query_to_df(query_inline)
                except Exception as e2:
                    st.error(f"Image search failed: {e2}")
        else:
            st.warning("Could not compute image embedding. Please try another image.")

    elif query_text.strip():
        embedding_col = "text_embedding"
        query = f"""
            SELECT base.product_id, base.title, base.categories, base.description, base.uri, distance
            FROM VECTOR_SEARCH(
              TABLE `{PRODUCTS_TABLE_ID}`,
              '{embedding_col}',
              (
                SELECT
                  ml_generate_embedding_result,
                  content AS query
                FROM ML.GENERATE_EMBEDDING(
                  MODEL `{TEXT_Embedding_MODEL_ID}`,
                  (SELECT @q AS content)
                )
              ),
              top_k => {int(top_k)}
            )
            ORDER BY distance ASC
        """
        try:
            results_df = run_query_to_df(
                {
                    "sql": query,
                    "params": [{"name": "q", "type": "STRING", "value": query_text}],
                }
            )
        except Exception:
            try:
                safe_q = query_text.replace('"', '\\"')
                query_fallback = query.replace("@q", f'"{safe_q}"')
                results_df = run_query_to_df(query_fallback)
            except Exception as e2:
                st.error(f"Query failed: {e2}")
    else:
        st.warning("Please enter a text query or upload an image.")

# ---------- Display ----------
if results_df is None:
    st.markdown("### Ready to search")
else:
    if results_df.empty:
        st.info("No results found. Try a different query or image.")
    else:
        display_query = "uploaded image" if uploaded_img is not None else query_text
        st.markdown(f"### Results for: _{display_query}_")
        cols_per_row = 3
        rows = int(np.ceil(len(results_df) / cols_per_row))
        cards = results_df.to_dict(orient="records")

        for r in range(rows):
            cols = st.columns(cols_per_row)
            for i in range(cols_per_row):
                idx = r * cols_per_row + i
                if idx >= len(cards):
                    continue
                item = cards[idx]
                with cols[i]:
                    with st.container(border=True):
                        uri = item.get("uri")
                        uri = uri.replace("gs://", "https://storage.googleapis.com/", 1)
                        if isinstance(uri, str) and uri.startswith("http"):
                            st.image(uri, use_container_width=True)

                        st.markdown(f"**{item.get('title','')}**")

                        desc = item.get("description", "")
                        if isinstance(desc, (list, tuple, dict, np.ndarray)):
                            text = str(desc)
                        else:
                            if desc is None:
                                text = ""
                            else:
                                try:
                                    is_nan = pd.isna(desc)
                                except Exception:
                                    is_nan = False
                                text = "" if (isinstance(is_nan, (bool, np.bool_)) and is_nan) else str(desc)

                        if text.strip():
                            st.write(text[:200] + ("…" if len(text) > 200 else ""))

                        uid = f"{item.get('product_id','')}_{idx}"
                        st.button("Add to shortlist", key=f"add_{uid}")
                        st.button("Product details", key=f"details_{uid}")

st.divider()
st.caption("© 2025 Smart Store • Built with Streamlit and BigQuery AI")

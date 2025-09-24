# pages/3_Dashboard.py
"""
Support Dashboard (Manager View)
- Shows real-time insights generated directly in BigQuery via AI functions.
- Tickets: sentiment KPIs/distribution, priority mix, top types, churn risk (avg, p90, buckets, top risky).
- Products: abusive comments KPIs/distribution/examples, boolean satisfaction KPIs/distribution.

Requirements:
- `services.dashboard_helper` exposes SQL helpers that return pandas DataFrames.
- `config.py` must define PROJECT_ID and DATASET_ID.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import altair as alt

from config import PROJECT_ID, DATASET_ID
from services.dashboard_helper import (
    # Tickets
    kpis_tickets,
    dist_sentiment_tickets,
    priority_distribution_tickets,
    type_top5_tickets,
    # Products - abuse
    kpis_abuse_products,
    abuse_distribution_products,
    abusive_examples_products,
    # Products - satisfaction (BOOLEAN)
    satisfaction_bool_kpis_products,
    satisfaction_bool_distribution_products,
    # Churn
    churn_kpis_tickets,
    churn_distribution_tickets,
    churn_top_tickets,
)

# -------------------------- Page config --------------------------
st.set_page_config(page_title="Support Dashboard", page_icon="📊", layout="wide")
st.title("📊 Support Dashboard")
st.caption("Real-time insights powered by BigQuery AI, computed on your data.")

# -------------------------- Helpers --------------------------
def safe_float(x, default=0.0) -> float:
    """
    Convert any value to float safely.
    Returns `default` if value is None, NaN, or cannot be cast to float.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    try:
        return float(x)
    except Exception:
        return default


# -------------------------- Tabs --------------------------
tab_tickets, tab_products = st.tabs(["🎟️ Tickets", "🛒 Products"])


# ============================== TICKETS TAB ==============================
with tab_tickets:
    st.subheader("Tickets Overview")

    # --- KPIs: total, negative rate, urgent ---
    kpis_df = kpis_tickets(PROJECT_ID, DATASET_ID)
    if kpis_df.empty:
        total_tickets = 0
        negative_rate = 0.0
        urgent_count = 0
    else:
        r = kpis_df.iloc[0]
        total_tickets = int(r.get("total_tickets", 0) or 0)
        negative_rate = safe_float(r.get("negative_rate", 0.0))
        urgent_count = int(r.get("urgent_count", 0) or 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Tickets", f"{total_tickets:,}", help="Number of tickets scored in this view.")
    with c2:
        st.metric("Negative Rate", f"{negative_rate:.1%}", help="Share of tickets with negative sentiment.")
    with c3:
        st.metric("Urgent Tickets", f"{urgent_count:,}", help="Count of tickets labeled high/urgent/P1/critical.")

    st.markdown("---")

    # --- Charts data fetch ---
    dist_sent = dist_sentiment_tickets(PROJECT_ID, DATASET_ID)
    priority_df = priority_distribution_tickets(PROJECT_ID, DATASET_ID)
    type_df = type_top5_tickets(PROJECT_ID, DATASET_ID)

    colA, colB = st.columns(2)

    # 1) Sentiment Distribution (Tickets)
    colA.subheader("Sentiment Distribution")
    if not dist_sent.empty and {"sentiment", "cnt"}.issubset(dist_sent.columns):
        chart_sent = (
            alt.Chart(dist_sent)
            .mark_bar()
            .encode(
                x=alt.X("sentiment:N", title="Sentiment", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("cnt:Q", title="Count"),
                tooltip=[
                    alt.Tooltip("sentiment:N", title="Sentiment"),
                    alt.Tooltip("cnt:Q", title="Count"),
                ],
            )
        ).properties(height=300)
        colA.altair_chart(chart_sent, use_container_width=True)
    else:
        colA.info("No data to display for sentiment distribution.")

    # 2) Priority Distribution (%) (Tickets)
    colB.subheader("Priority Distribution (%)")
    if not priority_df.empty and {"priority", "pct"}.issubset(priority_df.columns):
        chart_prio = (
            alt.Chart(priority_df)
            .mark_bar()
            .encode(
                x=alt.X("priority:N", title="Priority", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("pct:Q", title="Percentage", axis=alt.Axis(format="%")),
                tooltip=[
                    alt.Tooltip("priority:N", title="Priority"),
                    alt.Tooltip("pct:Q", title="Percentage", format=".1%"),
                ],
            )
        ).properties(height=300)
        colB.altair_chart(chart_prio, use_container_width=True)
    else:
        colB.info("No data to display for priority distribution.")

    st.markdown("---")

    # 3) Ticket Classification (Top 5)
    st.subheader("Ticket Classification (Top 5)")
    if not type_df.empty and {"type", "cnt"}.issubset(type_df.columns):
        order = type_df.sort_values("cnt", ascending=False)["type"].tolist()
        chart_type = (
            alt.Chart(type_df)
            .mark_bar()
            .encode(
                x=alt.X("type:N", title="Type", sort=order, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("cnt:Q", title="Count"),
                tooltip=[
                    alt.Tooltip("type:N", title="Type"),
                    alt.Tooltip("cnt:Q", title="Count"),
                ],
            )
        ).properties(height=300)
        st.altair_chart(chart_type, use_container_width=True)
    else:
        st.info("No data to display for ticket classification.")

    st.markdown("---")
    st.subheader("Churn Risk (Tickets)")

    # 4) KPIs: churn ---
    churn_kpis = churn_kpis_tickets(PROJECT_ID, DATASET_ID, high_threshold=0.7)
    avg_risk = p90_risk = 0.0
    high_risk_count = total_t = 0
    if not churn_kpis.empty:
        r = churn_kpis.iloc[0]
        avg_risk = safe_float(r.get("avg_risk", 0.0))
        p90_risk = safe_float(r.get("p90_risk", 0.0))
        high_risk_count = int(r.get("high_risk_count", 0) or 0)
        total_t = int(r.get("total_tickets", 0) or 0)

    c_r1, c_r2, c_r3, c_r4 = st.columns(4)
    c_r1.metric("Avg Risk", f"{avg_risk:.2f}", help="Average churn risk across scored tickets.")
    c_r2.metric("P90 Risk", f"{p90_risk:.2f}", help="90th percentile of churn risk.")
    c_r3.metric("High-Risk Tickets (≥0.7)", f"{high_risk_count:,}", help="Tickets above the 0.7 threshold.")
    c_r4.metric("Total Scored", f"{total_t:,}", help="Total tickets included in churn scoring.")

    # Distribution chart
    dist = churn_distribution_tickets(PROJECT_ID, DATASET_ID)
    st.caption("Risk Distribution")
    if not dist.empty and {"bucket_label", "cnt"}.issubset(dist.columns):
        chart_churn = (
            alt.Chart(dist)
            .mark_bar()
            .encode(
                x=alt.X("bucket_label:N", title="Risk Bucket", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("cnt:Q", title="Count"),
                tooltip=["bucket_label", "cnt"],
            )
        ).properties(height=300)
        st.altair_chart(chart_churn, use_container_width=True)
    else:
        st.info("No data to display for churn distribution.")

    # 5) Top risky tickets table
    top_risky = churn_top_tickets(PROJECT_ID, DATASET_ID, limit=20)
    st.caption("Top Risky Tickets")
    if not top_risky.empty:
        # Show also percentage for readability
        top_risky = top_risky.copy()
        top_risky["risk_pct"] = (top_risky["risk"].astype(float) * 100).round(1)
        st.dataframe(
            top_risky[["ticket_id", "risk", "risk_pct", "snippet"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No risky tickets found.")


# ============================== PRODUCTS TAB ==============================
with tab_products:
    st.subheader("Products Overview")

    # --- KPIs: Abuse (products comments) ---
    abuse_kpis = kpis_abuse_products(PROJECT_ID, DATASET_ID)
    abusive_count = 0
    abuse_rate = 0.0
    total_with_comments = 0
    if not abuse_kpis.empty:
        r2 = abuse_kpis.iloc[0]
        abusive_count = int(r2.get("abusive_count", 0) or 0)
        total_with_comments = int(r2.get("total_with_comments", 0) or 0)
        abuse_rate = safe_float(r2.get("abuse_rate", 0.0))

    # --- KPIs: Satisfaction (BOOLEAN) ---
    sat_kpis = satisfaction_bool_kpis_products(PROJECT_ID, DATASET_ID)
    satisfied_rate = 0.0
    satisfied_count = 0
    total_scored = 0
    if not sat_kpis.empty:
        r3 = sat_kpis.iloc[0]
        satisfied_count = int(r3.get("satisfied_count", 0) or 0)
        total_scored = int(r3.get("total_scored", 0) or 0)
        satisfied_rate = safe_float(r3.get("satisfaction_rate", 0.0))

    c4, c5, c6 = st.columns(3)
    c4.metric("Abusive Comments", f"{abusive_count:,}", help="Number of products whose comments were flagged as abusive.")
    c5.metric("Abuse Rate", f"{abuse_rate:.1%}", help="Share of flagged products among those with comments.")
    c6.metric("Satisfied Rate (Products)", f"{satisfied_rate:.1%}", help="Share of products classified as overall satisfied.")

    c7, c8 = st.columns(2)
    c7.metric("Satisfied Products", f"{satisfied_count:,}", help="Count of products with positive overall satisfaction.")
    c8.metric("Scored Products", f"{total_scored:,}", help="Total products considered for satisfaction scoring.")

    st.markdown("---")

    # --- Charts data (Products) ---
    abuse_dist = abuse_distribution_products(PROJECT_ID, DATASET_ID)
    examples_df = abusive_examples_products(PROJECT_ID, DATASET_ID, limit=20)
    sat_bool_dist = satisfaction_bool_distribution_products(PROJECT_ID, DATASET_ID)

    # Satisfaction (BOOLEAN) Distribution chart
    st.subheader("Customer Satisfaction (Products)")
    if not sat_bool_dist.empty and {"satisfied", "cnt"}.issubset(sat_bool_dist.columns):
        df = sat_bool_dist.copy()
        df["label"] = df["satisfied"].map({True: "Satisfied", False: "Not satisfied"})
        chart_sat = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("label:N", title="Satisfaction", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("cnt:Q", title="Count"),
                tooltip=[
                    alt.Tooltip("label:N", title="Satisfaction"),
                    alt.Tooltip("cnt:Q", title="Count"),
                ],
            )
        ).properties(height=300)
        st.altair_chart(chart_sat, use_container_width=True)
    else:
        st.info("No data to display for satisfaction.")

    st.markdown("---")

    # Content Moderation section
    st.subheader("Content Moderation (Product Comments)")
    col1, col2 = st.columns(2)

    # Abuse distribution chart
    col1.caption("Distribution of Abusive vs Non-Abusive")
    if not abuse_dist.empty and {"abusive", "cnt"}.issubset(abuse_dist.columns):
        abuse_dist = abuse_dist.copy()
        abuse_dist["label"] = abuse_dist["abusive"].map({True: "Abusive", False: "Clean"})
        chart_abuse = (
            alt.Chart(abuse_dist)
            .mark_bar()
            .encode(
                x=alt.X("label:N", title="Class", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("cnt:Q", title="Count"),
                tooltip=[
                    alt.Tooltip("label:N", title="Class"),
                    alt.Tooltip("cnt:Q", title="Count"),
                ],
            )
        ).properties(height=300)
        col1.altair_chart(chart_abuse, use_container_width=True)
    else:
        col1.info("No data to display for abuse distribution.")

    # Abusive examples table
    col2.caption("Examples (Flagged as Abusive)")
    if not examples_df.empty:
        col2.dataframe(
            examples_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        col2.info("No abusive comments found.")

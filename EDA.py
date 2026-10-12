"""Streamlit dashboard adapted from EDA.ipynb.

Run: python -m streamlit run EDA.py
Keep olist_orders_dataset.csv next to this file.
"""

from datetime import date
from pathlib import Path

import matplotlib

# Render plots without a desktop GUI, for Streamlit's worker thread and Cloud.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

TEAM_NAME = "team 10 of fin"
DATA_FILE = Path(__file__).with_name("olist_orders_dataset.csv")
DATASET_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
DATE_COLUMNS = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]
STAGES = {
    "approval_days": ("order_purchase_timestamp", "order_approved_at"),
    "handoff_days": ("order_approved_at", "order_delivered_carrier_date"),
    "transit_days": ("order_delivered_carrier_date", "order_delivered_customer_date"),
}
STAGE_LABELS = ["Purchase to approval", "Approval to handoff", "Handoff to delivery"]


@st.cache_data(show_spinner="Loading order data...")
def load_orders(path: str, file_version: int):
    """The file modification time invalidates cached data when the CSV changes."""
    raw = pd.read_csv(path, dtype={"order_id": "str", "customer_id": "str"})
    required = {"order_id", "customer_id", "order_status", *DATE_COLUMNS}
    if not required.issubset(raw.columns):
        raise ValueError("Missing columns: " + ", ".join(sorted(required.difference(raw.columns))))
    orders = raw.copy()
    parse_failures = {}
    for column in DATE_COLUMNS:
        orders[column] = pd.to_datetime(orders[column], errors="coerce")
        parse_failures[column] = int((raw[column].notna() & orders[column].isna()).sum())
    orders["delivery_days"] = (
        orders.order_delivered_customer_date - orders.order_purchase_timestamp
    ).dt.total_seconds() / 86400
    # Estimated timestamps are midnight: compare calendar dates, not timestamps.
    orders["late_days"] = (
        orders.order_delivered_customer_date.dt.normalize()
        - orders.order_estimated_delivery_date.dt.normalize()
    ).dt.days
    orders["eligible_delivery"] = (
        orders.order_status.eq("delivered") & orders.order_purchase_timestamp.notna()
        & orders.order_delivered_customer_date.notna()
        & orders.order_estimated_delivery_date.notna() & orders.delivery_days.ge(0)
    )
    orders["late"] = orders.late_days.gt(0)
    for name, (start, end) in STAGES.items():
        orders[name] = (orders[end] - orders[start]).dt.total_seconds() / 86400
    return orders, parse_failures


def select_cohort(orders, start, end):
    """Include every purchase on the end date, including late-night orders."""
    return orders.loc[
        orders.order_purchase_timestamp.ge(pd.Timestamp(start))
        & orders.order_purchase_timestamp.lt(pd.Timestamp(end) + pd.Timedelta(days=1))
    ].copy()


def monthly_delivery_metrics(valid, start, end):
    monthly = valid.groupby(valid.order_purchase_timestamp.dt.to_period("M")).agg(
        eligible_deliveries=("order_id", "nunique"), late_deliveries=("late", "sum"),
    ).reindex(pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M"))
    # Missing months remain NaN, so they appear as gaps rather than 0%.
    monthly["late_rate_pct"] = monthly.late_deliveries / monthly.eligible_deliveries * 100
    return monthly


def show_trend(monthly):
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    positions = list(range(len(monthly)))
    ax.plot(positions, monthly.late_rate_pct, color="#a04e24", marker="o")
    step = max(1, (len(monthly) + 11) // 12)
    ticks = positions[::step]
    ax.set_xticks(ticks, [str(monthly.index[i]) for i in ticks], rotation=45, ha="right")
    ax.set_xlabel("Purchase month")
    ax.set_ylabel("Late deliveries (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)


def show_stage_comparison(stage_data):
    stage_data = stage_data.copy()
    stage_data["delivery_result"] = stage_data.late.map({False: "On time", True: "Late"})
    grouped = stage_data.groupby("delivery_result")
    medians, counts = grouped[list(STAGES)].median(), grouped.size()
    outcomes = [name for name in ["On time", "Late"] if name in medians.index]
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    width = 0.8 / len(outcomes)
    for index, outcome in enumerate(outcomes):
        positions = [i - 0.4 + width / 2 + index * width for i in range(3)]
        bars = ax.bar(
            positions, medians.loc[outcome].values, width=width,
            color="#27658a" if outcome == "On time" else "#a04e24",
            hatch=None if outcome == "On time" else "//",
            label=f"{outcome} (n={counts[outcome]:,})",
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(range(3), STAGE_LABELS)
    ax.set_ylabel("Median duration (days)")
    ax.set_ylim(bottom=0)
    ax.margins(y=0.15)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.set_page_config(page_title="Olist Order Fulfillment", layout="wide")
    st.title(f"Olist Order Fulfillment Dashboard — {TEAM_NAME}")
    st.caption("For the fulfillment operations manager to identify delivery delays and prioritize investigation of handoff and transit stages.")
    try:
        orders, parse_failures = load_orders(str(DATA_FILE), DATA_FILE.stat().st_mtime_ns)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load order data: {exc}")
        st.info("Place olist_orders_dataset.csv in the same folder as EDA.py.")
        st.stop()
    purchase_dates = orders.order_purchase_timestamp.dropna()
    if purchase_dates.empty:
        st.error("No valid purchase dates are available.")
        st.stop()
    minimum, maximum = purchase_dates.min().date(), purchase_dates.max().date()
    st.caption(
        f"Data: orders purchased {minimum:%b %d, %Y}–{maximum:%b %d, %Y}, Kaggle snapshot. "
        "Production refresh: hourly; manager review: twice daily."
    )
    st.caption("This app displays a fixed historical CSV. Production refresh is the proposed business schedule.")
    st.sidebar.header("Purchase cohort")
    selected = st.sidebar.date_input(
        "Purchase date range",
        value=(min(max(date(2017, 1, 1), minimum), maximum), min(max(date(2018, 8, 31), minimum), maximum)),
        min_value=minimum, max_value=maximum, format="YYYY-MM-DD",
    )
    st.sidebar.caption("Default: Jan 2017–Aug 2018. Sparse edge months are available for inspection.")
    st.sidebar.markdown(f"[Dataset on Kaggle]({DATASET_URL})")
    if not isinstance(selected, (tuple, list)) or len(selected) != 2:
        st.info("Select both a start date and an end date.")
        st.stop()
    start, end = selected
    if start > end:
        st.warning("The start date must not be after the end date.")
        st.stop()

    cohort = select_cohort(orders, start, end)
    valid = cohort.loc[cohort.eligible_delivery].copy()
    late_count = int(valid.late.sum())
    columns = st.columns(3)
    columns[0].metric("Total Orders", f"{cohort.order_id.nunique():,}")
    columns[1].metric("Late Delivery Rate", f"{valid.late.mean():.2%}" if len(valid) else "N/A")
    columns[2].metric("Median Delivery Time", f"{valid.delivery_days.median():.2f} days" if len(valid) else "N/A")
    st.caption(
        f"Purchase cohort: {start:%Y-%m-%d}–{end:%Y-%m-%d}. Eligible delivered orders: {len(valid):,}; late orders: {late_count:,}. "
        "Total Orders includes all statuses. Delivery metrics include eligible delivered orders only. "
        "Late means delivered after the promised calendar date."
    )
    if cohort.empty:
        st.info("No orders were purchased in this date range. Choose another range.")
        st.stop()

    st.subheader("Late delivery rate by purchase month")
    if valid.empty:
        st.info("No eligible delivered orders are available. Delivery metrics and charts cannot be calculated.")
    else:
        monthly = monthly_delivery_metrics(valid, start, end)
        show_trend(monthly)
        st.caption("Each point uses observed delivery outcomes of orders purchased in that month. Months without eligible deliveries appear as gaps. Partial months follow the selected date range.")
        with st.expander("Monthly sample sizes and rates"):
            table = monthly.rename_axis("Purchase month").reset_index()
            table["Purchase month"] = table["Purchase month"].astype(str)
            st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Where does delivery take longer?")
    stage_data = valid.loc[valid[list(STAGES)].ge(0).all(axis=1)].copy()
    if stage_data.empty:
        st.info("No orders have complete, nonnegative timestamps for all three stages.")
    else:
        show_stage_comparison(stage_data)
        st.caption(
            f"Common valid stage sample: {len(stage_data):,} orders; {len(valid) - len(stage_data):,} eligible deliveries excluded for missing or negative stage durations. "
            "Stage medians cannot be added into an overall median. These comparisons guide investigation; they do not establish the cause of delay or identify a responsible carrier."
        )

    with st.expander(f"Late order details ({late_count:,} orders)"):
        details = valid.loc[valid.late, [
            "order_id", "order_purchase_timestamp", "order_estimated_delivery_date",
            "order_delivered_customer_date", "late_days", "delivery_days",
        ]].sort_values("late_days", ascending=False)
        if details.empty:
            st.info("No observed late deliveries in this purchase cohort.")
        else:
            st.dataframe(details.head(500), hide_index=True, width="stretch")
            st.caption("Showing up to 500 orders, sorted by delay. Download includes all selected late orders.")
            st.download_button(
                "Download late orders CSV", data=details.to_csv(index=False).encode("utf-8"),
                file_name=f"olist_late_orders_{start}_{end}.csv", mime="text/csv",
            )

    with st.expander("Data quality and interpretation"):
        delivered = cohort.loc[cohort.order_status.eq("delivered")]
        missing_actual = int(delivered.order_delivered_customer_date.isna().sum())
        inconsistent = int((cohort.order_status.ne("delivered") & cohort.order_delivered_customer_date.notna()).sum())
        st.write(f"In this cohort, {missing_actual:,} delivered orders have no actual delivery timestamp, and {inconsistent:,} orders have an actual delivery timestamp but another status. They are not included in eligible delivery metrics.")
        stage_quality = pd.DataFrame([
            {"Stage": label, "Missing timestamps": int(delivered[name].isna().sum()), "Negative duration": int(delivered[name].lt(0).sum())}
            for name, label in zip(STAGES, STAGE_LABELS)
        ])
        st.dataframe(stage_quality, hide_index=True, width="stretch")
        st.write("Sparse edge months and unobserved months do not prove a business collapse. Delivery times have a long tail; original observations are retained, and the headline uses the median. Undelivered orders are absent from completed-delivery metrics, so these rates are not the final delay probability of every purchased order. Source timestamps have no explicit timezone.")
        st.write("This table cannot show revenue, profit, region, carrier rankings or repeat-customer rates. It also lacks status history and an explicit extraction time: historical undelivered orders are not presented as today's overdue backlog.")
        st.caption(f"Source records: {len(orders):,}. Duplicate order IDs: {int(orders.order_id.duplicated().sum()):,}. Unparseable nonempty timestamps: {sum(parse_failures.values()):,}.")


if __name__ == "__main__":
    main()

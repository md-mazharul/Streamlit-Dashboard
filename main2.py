# app.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Movie Ratings", layout="wide")
st.title("🎬 Movie Ratings — Age & Genre Explorer")

# -----------------------------
# Load data
# -----------------------------
# Tries: 1) user upload, 2) your Windows path, 3) workspace upload path
default_paths = [
    Path(r"C:\Users\omibr\Downloads\movie_ratings.csv"),
    Path("/mnt/data/movie_ratings.csv"),
]
#uploaded = st.file_uploader("Upload movie_ratings.csv (optional)", type=["csv"])

@st.cache_data(show_spinner=False)
def load_csv(file_or_path):
    if hasattr(file_or_path, "read"):
        return pd.read_csv(file_or_path)
    return pd.read_csv(file_or_path)

df = None
for p in default_paths:
    if p.exists():
        df = load_csv(p)
        break

if df is None:
    st.error("CSV not found. Upload the file or update the path at the top.")
    st.stop()

#st.caption("Data preview")
#st.dataframe(df.head())

# -----------------------------
# Column detection
# -----------------------------
def find_col(df, names):
    for want in names:
        if want in df.columns:
            return want
        for c in df.columns:
            if c.lower() == want.lower():
                return c
    return None

age_col    = find_col(df, ["age", "user_age", "age_group"])
rating_col = find_col(df, ["rating", "score", "stars"])
genre_col  = find_col(df, ["genre", "genres"])

year_col   = find_col(df, ["year", "release_year", "movie_year"])
timestamp_col  = find_col(df, ["title", "movie_title", "name"])

# if no year but we have a timestamp, derive year
if year_col is None and timestamp_col:
    df["year"] = pd.to_datetime(df[timestamp_col], errors="coerce").dt.year
    year_col = "year"


if age_col is None:
    st.error("No age column found. Include a column like: age / user_age / age_group.")
    st.stop()

# Clean types
df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
df = df.dropna(subset=[age_col])

if rating_col:
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")

# Split pipe-separated genres: "Action|Comedy"
if genre_col and df[genre_col].dtype == object and df[genre_col].str.contains(r"\|").any():
    df = df.assign(**{genre_col: df[genre_col].fillna("").str.split("|")}).explode(genre_col)
    df[genre_col] = df[genre_col].str.strip()
    df = df[df[genre_col] != ""]

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

# Age bins
default_bins = [0, 12, 18, 25, 35, 45, 55, 65, 75, 120]
bins_text = st.sidebar.text_input("Age bins (comma-separated)", ", ".join(map(str, default_bins)))
try:
    bin_edges = [int(x.strip()) for x in bins_text.split(",") if x.strip()]
    if len(bin_edges) < 3:
        raise ValueError
except Exception:
    st.sidebar.error("Invalid bins. Using defaults.")
    bin_edges = default_bins

age_labels = [f"{bin_edges[i]}–{bin_edges[i+1]-1}" for i in range(len(bin_edges)-1)]
df["age_bin"] = pd.cut(df[age_col], bins=bin_edges, right=False, labels=age_labels, include_lowest=True)

metric = st.sidebar.radio("Heatmap metric", ["Average rating", "Number of ratings"])

# Genre multi-select (global filter)
if genre_col:
    all_genres = sorted([g for g in df[genre_col].dropna().unique() if str(g).strip()])
    selected_genres = st.sidebar.multiselect("Filter by genre (applies to all sections)", all_genres)
    if selected_genres:
        df = df[df[genre_col].isin(selected_genres)]
# --- Year filter in sidebar ---

if year_col and df[year_col].notna().any():
    y_min = int(df[year_col].min())
    y_max = int(df[year_col].max())
    year_range = st.sidebar.slider("Filter by year", y_min, y_max, (y_min, y_max), step=1)
    df = df[(df[year_col] >= year_range[0]) & (df[year_col] <= year_range[1])]

# -----------------------------
# Tabs: Overview | Genre | Heatmap
# -----------------------------
tab_overview, tab_genre, tab_heatmap, tab_year = st.tabs(["Overview", "🎭 Genre", "🔥 Heatmap", "📅 Year"])

with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        # Age distribution
        st.subheader("Age Distribution (Counts)")
        age_counts = df["age_bin"].value_counts().reindex(age_labels).fillna(0).reset_index()
        age_counts.columns = ["age_bin", "Count"]
        fig_age = px.bar(age_counts, x="age_bin", y="Count", title=None)
        st.plotly_chart(fig_age, use_container_width=True)

    with c2:
        # Ratings distribution
        st.subheader("Rating Distribution")
        if rating_col and df[rating_col].notna().any():
            fig_r = px.histogram(df, x=rating_col, nbins=20, title=None)
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("No numeric rating column found.")

with tab_genre:
    st.subheader("Genre Insights")

    if not genre_col:
        st.warning("No genre column found. Add a `genre` or `genres` column to enable this section.")
    else:
        # 1) Genre distribution (counts)
        st.markdown("**Ratings per Genre**")
        genre_counts = (df.groupby(genre_col)[age_col]
                        .count()
                        .sort_values(ascending=False)
                        .rename("NumRatings")
                        .reset_index())
        fig_gc = px.bar(genre_counts, x="NumRatings", y=genre_col, orientation="h", title=None)
        st.plotly_chart(fig_gc, use_container_width=True)

        # 2) Average rating by genre
        st.markdown("**Average Rating by Genre**")
        if rating_col and df[rating_col].notna().any():
            genre_avg = (df.groupby(genre_col)[rating_col]
                         .mean()
                         .sort_values(ascending=False)
                         .round(2)
                         .rename("AvgRating")
                         .reset_index())
            fig_ga = px.bar(genre_avg, x="AvgRating", y=genre_col, orientation="h", title=None)
            st.plotly_chart(fig_ga, use_container_width=True)
        else:
            st.info("No numeric rating column to compute averages.")

        # 3) Table: Top N genres
        st.markdown("**Top Genres Table**")
        if rating_col and df[rating_col].notna().any():
            top_tbl = (df.groupby(genre_col)
                         .agg(NumRatings=(age_col, "count"),
                              AvgRating=(rating_col, "mean"))
                         .sort_values("NumRatings", ascending=False)
                         .round({"AvgRating": 2})
                         .reset_index())
        else:
            top_tbl = (df.groupby(genre_col)
                         .agg(NumRatings=(age_col, "count"))
                         .sort_values("NumRatings", ascending=False)
                         .reset_index())

        st.dataframe(top_tbl, use_container_width=True)

with tab_heatmap:
    st.subheader("Age × Genre Heatmap")

    def build_heatmap(df):
        if genre_col and df[genre_col].notna().any():
            if metric == "Average rating" and rating_col and df[rating_col].notna().any():
                pv = df.pivot_table(index=genre_col, columns="age_bin", values=rating_col, aggfunc="mean")
                title = "Average Rating by Genre × Age"
                z = "Avg rating"
            else:
                pv = df.pivot_table(index=genre_col, columns="age_bin", values=age_col, aggfunc="count")
                title = "Number of Ratings by Genre × Age"
                z = "Count"
            return pv, title, z

        # Fallbacks (no genre)
        if rating_col and df[rating_col].notna().any():
            valid = df[rating_col].dropna()
            rmin, rmax = float(valid.min()), float(valid.max())
            edges = np.arange(0, 5.5, 0.5) if (0 <= rmin <= 5 and rmax <= 5) else np.linspace(rmin, rmax, 11)
            df["_rating_bin"] = pd.cut(df[rating_col], bins=edges, include_lowest=True)
            pv = df.pivot_table(index="_rating_bin", columns="age_bin", values=rating_col, aggfunc="count")
            return pv, "Number of Ratings by Rating Bin × Age", "Count"

        pv = df.pivot_table(index="age_bin", values=age_col, aggfunc="count")
        pv.columns = ["Count"]
        return pv, "Counts by Age Bin", "Count"

    pv, title, zlabel = build_heatmap(df)

    # Ensure nice labels
    if isinstance(pv.index, pd.IntervalIndex): pv.index = pv.index.astype(str)
    if isinstance(pv.columns, pd.IntervalIndex): pv.columns = pv.columns.astype(str)

    # 1D fallback to bar
    if list(pv.columns) == ["Count"] and pv.index.name == "age_bin":
        fig = px.bar(pv.reset_index(), x="age_bin", y="Count", title=title)
    else:
        fig = px.imshow(
            pv,
            aspect="auto",
            labels=dict(x="Age bin", y=pv.index.name or "Row", color=zlabel),
            color_continuous_scale="Viridis",
            origin="upper",
        )
        fig.update_layout(xaxis_title="Age bin", yaxis_title=pv.index.name or "Category")

    st.plotly_chart(fig, use_container_width=True)

    # Optional: download pivot
    st.download_button(
        "Download heatmap data (CSV)",
        pv.to_csv().encode("utf-8"),
        "heatmap_pivot.csv",
        "text/csv"
    )


with tab_year:
    st.subheader("Year Insights")

    if not year_col or df[year_col].isna().all():
        st.info("No year information available.")
    else:
        c1, c2 = st.columns(2)

        # Ratings per Year (count)
        with c1:
            yr_counts = (df.groupby(year_col)[age_col]
                         .count()
                         .reset_index(name="NumRatings"))
            fig_yc = px.bar(yr_counts, x=year_col, y="NumRatings",
                            title="Number of Ratings per Year")
            st.plotly_chart(fig_yc, use_container_width=True)

        # Average Rating by Year (if ratings exist)
        with c2:
            if rating_col and df[rating_col].notna().any():
                yr_avg = (df.groupby(year_col)[rating_col]
                          .mean()
                          .reset_index()
                          .rename(columns={rating_col: "AvgRating"}))
                fig_ya = px.line(yr_avg, x=year_col, y="AvgRating", markers=True,
                                 title="Average Rating by Year")
                st.plotly_chart(fig_ya, use_container_width=True)
            else:
                st.info("No numeric rating column to compute averages by year.")

        # Optional: Decade summary
        if "decade" in df.columns:
            st.markdown("**Ratings per Decade**")
            dec_counts = (df.groupby("decade")[age_col]
                          .count()
                          .reset_index(name="NumRatings"))
            fig_dec = px.bar(dec_counts, x="decade", y="NumRatings", title=None)
            st.plotly_chart(fig_dec, use_container_width=True)



from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"

st.set_page_config(page_title="V1 Manifold Explorer", layout="wide")
st.title("Latent manifold geometry of V1 population codes")
st.caption("Allen Brain Observatory natural movie calcium-imaging analysis")

embedding_files = sorted(PROCESSED.glob("session_*_embeddings.npz"))
if not embedding_files:
    st.info(
        "No generated embeddings were found yet. I need to run the notebooks or `make manifold` first. "
        "The app is intentionally read-only and does not fabricate example neural data."
    )
    st.stop()

session_file = st.sidebar.selectbox("Session embedding file", embedding_files, format_func=lambda p: p.name)
data = np.load(session_file)
embedding_names = [k for k in data.files if k != "frame"]
embedding_name = st.sidebar.selectbox("Embedding", embedding_names)
Z = data[embedding_name]
frame = data["frame"] if "frame" in data.files else np.arange(Z.shape[0])

labels_file = PROCESSED / session_file.name.replace("_embeddings.npz", "_frame_features.csv")
if labels_file.exists():
    labels = pd.read_csv(labels_file)
else:
    labels = pd.DataFrame({"movie_frame": frame})

plot_df = pd.DataFrame({
    "x": Z[:, 0],
    "y": Z[:, 1],
    "z": Z[:, 2] if Z.shape[1] > 2 else np.zeros(Z.shape[0]),
    "movie_frame": frame,
})
plot_df = plot_df.merge(labels, on="movie_frame", how="left")
color_options = [c for c in ["movie_frame", "orientation_bin", "spatial_frequency_bin", "population_l2_norm"] if c in plot_df.columns]
color = st.sidebar.selectbox("Colour trajectory by", color_options)

fig = px.scatter_3d(plot_df, x="x", y="y", z="z", color=color, hover_data=["movie_frame"], opacity=0.85)
fig.update_traces(marker=dict(size=3))
fig.update_layout(height=650, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Generated metric tables")
for table in sorted(TABLES.glob("*.csv")):
    with st.expander(table.name):
        st.dataframe(pd.read_csv(table), use_container_width=True)

st.subheader("Generated figures")
figures = sorted(FIGURES.glob("*.png"))
if not figures:
    st.write("No generated figures found yet.")
else:
    cols = st.columns(2)
    for i, fig_path in enumerate(figures):
        cols[i % 2].image(str(fig_path), caption=fig_path.name, use_container_width=True)

"""Accessible Plotly visualizations for CodeShield review dashboard."""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd


# Sequential hue (one-color magnitude). Keep it simple and colorblind-safe.
SEQUENTIAL_BLUE = [
    [0.0, "#F1F5F9"],
    [0.25, "#CBDCEB"],
    [0.5, "#7EA6C8"],
    [0.75, "#3976A8"],
    [1.0, "#123B5D"],
]


def _document_labels(documents: list) -> list[str]:
    return [doc.get("filename", f"Document {i + 1}") for i, doc in enumerate(documents)]


def create_similarity_heatmap(documents: list, similarity_matrix: np.ndarray) -> go.Figure:
    """Pairwise similarity matrix (0..1) with clear hover and percent annotations."""

    doc_names = _document_labels(documents)
    matrix = np.clip(np.asarray(similarity_matrix, dtype=float), 0, 1)

    fig = px.imshow(
        matrix,
        x=doc_names,
        y=doc_names,
        color_continuous_scale=SEQUENTIAL_BLUE,
        zmin=0,
        zmax=1,
        text_auto=".0%",
        aspect="auto",
        labels={"color": "Similarity"},
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b> ↔ <b>%{x}</b><br>"
            "Similarity: %{z:.1%}<extra></extra>"
        ),
        xgap=2,
        ygap=2,
    )

    fig.update_layout(
        title=dict(text="Similarity Heatmap"),
        xaxis_title="Submission",
        yaxis_title="Submission",
        template="plotly_white",
        coloraxis_colorbar=dict(title="Similarity", tickformat=".0%"),
        margin=dict(l=20, r=20, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-35, automargin=True)
    fig.update_yaxes(automargin=True)

    return fig


def create_feature_barchart(features: dict) -> go.Figure:
    """Horizontal bar chart for similarity signal components."""

    key_labels = {
        "lexical_similarity": "Lexical",
        "semantic_similarity": "Semantic",
        "structural_similarity": "Structural",
        "word_overlap": "Word overlap",
        "phrase_overlap": "Phrase overlap",
        "ink_similarity": "Ink",
        "character_size_similarity": "Character size",
        "line_spacing_similarity": "Line spacing",
        "word_spacing_similarity": "Word spacing",
        "slant_similarity": "Slant",
        "layout_similarity": "Layout",
        "handwriting_similarity": "Handwriting",
    }

    # Keep a stable order (important for reviewers).
    feature_order = list(key_labels.keys())

    rows = []
    for k in feature_order:
        v = features.get(k)
        if v is None:
            continue
        rows.append({"Feature": key_labels.get(k, k), "Similarity": float(v)})

    df = pd.DataFrame(rows)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No comparable features available", showarrow=False)
        return fig

    fig = px.bar(
        df,
        x="Similarity",
        y="Feature",
        orientation="h",
        color="Similarity",
        color_continuous_scale=SEQUENTIAL_BLUE,
        range_color=[0, 1],
        labels={"Similarity": "Similarity", "Feature": "Signal"},
    )

    fig.update_traces(
        text=df["Similarity"].map(lambda x: f"{x:.0%}"),
        texttemplate="%{text}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Similarity: %{x:.1%}<extra></extra>",
    )

    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor="#E2E8F0")
    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        title=dict(text="Similarity Signal Breakdown"),
        template="plotly_white",
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=20, r=55, t=60, b=30),
        height=350,
    )

    return fig

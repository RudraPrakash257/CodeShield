"""Network graph visualization for flagged similarity pairs."""

import plotly.graph_objects as go
import networkx as nx
import numpy as np


def create_similarity_network(
    documents: list,
    similarity_scores: list,
    threshold: float = 0.7,
) -> go.Figure:
    """Nodes=submissions, edges=flagged pairs (score >= threshold)."""

    G = nx.Graph()

    for i, doc in enumerate(documents):
        G.add_node(i, label=doc.get("filename", f"Doc {i + 1}"))

    for i, j, score in similarity_scores:
        if score >= threshold:
            G.add_edge(i, j, weight=float(score))

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Edges
    edge_x = []
    edge_y = []
    edge_hover = []
    for i, j in G.edges():
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_hover.append(f"{G.nodes[i]['label']} ↔ {G.nodes[j]['label']}: {G.edges[i, j]['weight']:.1%}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#9AA4B2"),
        hoverinfo="none",
        showlegend=False,
    )

    # Nodes
    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]["label"])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hovertemplate="<b>%{text}</b><extra></extra>",
        text=node_text,
        textposition="top center",
        hoverlabel=dict(bgcolor="white"),
        marker=dict(
            size=18,
            color="#1F77B4",
            line=dict(width=2, color="#0B2239"),
            opacity=0.95,
        ),
        showlegend=False,
    )

    # Optional: display message if no edges
    layout_title = "Similarity Network (flagged edges)" if G.number_of_edges() > 0 else "Similarity Network (no flagged edges)"

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Similarity Network",
                font=dict(
                    size=16,
                    color="white"
                ),
            ),
            template="plotly_dark",
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=45),
            xaxis=dict(
                title=dict(
                    text="Documents",
                    font=dict(size=14)
                    )
                ),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return fig

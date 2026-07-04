"""
visualization/dashboard.py

A minimal interactive dashboard (Plotly + Dash) for exploring results by
model/language/category. Run with: python visualization/dashboard.py
"""

from __future__ import annotations
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output


def build_app(csv_path: str = "results/csv/refusal_rate.csv") -> Dash:
    df = pd.read_csv(csv_path, index_col=0).reset_index().melt(
        id_vars="model", var_name="language", value_name="refusal_rate"
    )

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("SentinelAI — Cross-Lingual Safety Dashboard"),
            dcc.Dropdown(
                id="model-filter",
                options=[{"label": m, "value": m} for m in df["model"].unique()],
                value=list(df["model"].unique()),
                multi=True,
            ),
            dcc.Graph(id="refusal-graph"),
        ]
    )

    @app.callback(Output("refusal-graph", "figure"), Input("model-filter", "value"))
    def update_graph(selected_models):
        filtered = df[df["model"].isin(selected_models)]
        fig = px.bar(
            filtered,
            x="language",
            y="refusal_rate",
            color="model",
            barmode="group",
            title="Refusal Rate (%) by Language",
        )
        return fig

    return app


if __name__ == "__main__":
    app = build_app()
    app.run(debug=True)

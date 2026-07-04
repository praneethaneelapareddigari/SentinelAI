"""
visualization/plots.py

Generates the charts described in the project brief: safety-score-by-language
heatmap, refusal-rate bars, radar chart per model, all from the CSVs produced
by evaluation/scoring.py.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def heatmap(csv_path: str, out_path: str, title: str) -> None:
    df = pd.read_csv(csv_path, index_col=0)
    fig, ax = plt.subplots(figsize=(8, 4 + 0.3 * len(df)))
    im = ax.imshow(df.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def grouped_bar(csv_path: str, out_path: str, title: str, ylabel: str) -> None:
    df = pd.read_csv(csv_path, index_col=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    df.T.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(title="model")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def radar_chart(csv_path: str, out_path: str, title: str) -> None:
    df = pd.read_csv(csv_path, index_col=0)
    languages = list(df.columns)
    angles = np.linspace(0, 2 * np.pi, len(languages), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for model in df.index:
        values = df.loc[model].tolist()
        values += values[:1]
        ax.plot(angles, values, label=model, linewidth=2)
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(languages)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all(results_dir: str = "results/csv", out_dir: str = "reports/figures") -> None:
    heatmap(
        f"{results_dir}/refusal_rate.csv",
        f"{out_dir}/refusal_heatmap.png",
        "Refusal Rate (%) by Model x Language",
    )
    grouped_bar(
        f"{results_dir}/refusal_rate.csv",
        f"{out_dir}/refusal_bars.png",
        "Refusal Rate by Language",
        "Refusal rate (%)",
    )
    radar_chart(
        f"{results_dir}/refusal_rate.csv",
        f"{out_dir}/refusal_radar.png",
        "Safety Consistency Across Languages",
    )
    print(f"Figures written to {out_dir}/")


if __name__ == "__main__":
    generate_all()

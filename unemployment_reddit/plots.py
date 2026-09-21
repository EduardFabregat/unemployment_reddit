"""Figures used in the analysis. Each function returns the matplotlib figure."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

from .data import monthly_share


def comments_per_day(df: pd.DataFrame):
    daily = df.groupby("simple_date", as_index=False).agg({"comment": "count"})
    daily["simple_date"] = pd.to_datetime(daily["simple_date"])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=daily, x="simple_date", y="comment", color="teal", linewidth=2, ax=ax)
    ax.set_title("Number of Comments per Day", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Comment Count", fontsize=12)
    fig.tight_layout()
    return fig


def sample_vs_population(df: pd.DataFrame, df_docs: pd.DataFrame):
    """Line chart of monthly share of documents, population vs. the topic-model sample."""
    compare = pd.DataFrame(
        {"Population": monthly_share(df), "Sample": monthly_share(df_docs)}
    ).fillna(0)
    compare.index = compare.index.astype(str)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=compare, palette=["royalblue", "crimson"], linewidth=2, ax=ax)
    ax.set_title(
        "Sampling Verification: Population vs. Sample Temporal Distribution",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Proportion of Total Dataset")
    ax.set_xlabel("Timeline")
    ax.set_xticks(range(0, len(compare), 6))  # every 6th month keeps the axis readable
    ax.set_xticklabels(compare.index[::6], rotation=45)
    fig.tight_layout()
    return fig


def frame_count_curves(diagnostics: pd.DataFrame, best_k: int):
    """Silhouette and elbow curves used to choose the number of macro-frames."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(diagnostics["k"], diagnostics["silhouette"], marker="o", color="royalblue", linewidth=2)
    ax1.axvline(best_k, color="crimson", linestyle="--", label=f"Peak (K={best_k})")
    ax1.set_title("Silhouette Optimization Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Macro-Frames (K)")
    ax1.set_ylabel("Average Silhouette Score")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(diagnostics["k"], diagnostics["wcss"], marker="s", color="darkorange", linewidth=2)
    ax2.set_title("Elbow Method (Variance Explained)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Macro-Frames (K)")
    ax2.set_ylabel("Within-Cluster Variance (WCSS)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def topic_map(df_plot: pd.DataFrame, title: str = "Latent Macro-Framework of Discourse on the Unemployment Subreddit"):
    """Scatter of topics in 2-D UMAP space with a robust ellipse around each macro-frame."""
    from adjustText import adjust_text
    from sklearn.covariance import MinCovDet

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(20, 16))
    fig.patch.set_facecolor("#F0F0F0")
    ax.set_facecolor("#F0F0F0")

    n_frames = df_plot["cluster"].nunique()
    base_cmap = plt.colormaps["tab20"]
    colors = [base_cmap(i % 20) for i in range(n_frames)]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_frames)

    ax.scatter(
        df_plot["x"], df_plot["y"], c=df_plot["cluster"], cmap=cmap,
        s=120, alpha=0.8, edgecolors="w", zorder=3,
    )

    for cluster, color in zip(sorted(df_plot["cluster"].unique()), [cmap(i) for i in range(n_frames)]):
        points = df_plot.loc[df_plot["cluster"] == cluster, ["x", "y"]].to_numpy()
        if len(points) < 3:
            print(f"Macro-frame {cluster} has fewer than 3 topics; skipping its ellipse.")
            continue
        cov = MinCovDet(random_state=42).fit(points).covariance_
        eigvals, eigvecs = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 4 * np.sqrt(np.abs(eigvals))
        ax.add_patch(
            Ellipse(
                points.mean(axis=0), width, height, angle=angle, facecolor="none",
                edgecolor=color, linewidth=2.5, alpha=0.7, zorder=2,
            )
        )

    ax.set_title(title, fontsize=26, fontweight="bold", pad=25)
    ax.set_xlabel("UMAP Dimension 1", fontsize=14, fontweight="bold", alpha=0.6)
    ax.set_ylabel("UMAP Dimension 2", fontsize=14, fontweight="bold", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4, color="#D3D3D3")

    texts = [
        ax.text(row["x"], row["y"], row["topic_name"], fontsize=10, fontweight="medium", zorder=4)
        for _, row in df_plot.iterrows()
    ]
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.6, alpha=0.7))

    fig.tight_layout()
    return fig


def shap_summary(shap_values, sample: pd.DataFrame):
    """SHAP beeswarm of feature impact on cascade size."""
    import shap

    fig = plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values, sample, show=False)
    plt.title("SHAP Feature Impact on Conversation Cascade Size", fontsize=12)
    plt.tight_layout()
    return fig

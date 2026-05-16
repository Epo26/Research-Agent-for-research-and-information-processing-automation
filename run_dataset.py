import matplotlib
matplotlib.use("Agg")
import math
import mlflow
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from reserch_agent.graph import create_agent_graph
from reserch_agent.config import CONFIG, DATASET
from reserch_agent.metrics.evaluators import evaluate_all_metrics_super_judge
from reserch_agent.llm import llm_judge

matplotlib.rcParams["font.family"] = "DejaVu Sans"

logger = logging.getLogger(__name__)

METRIC_NAMES = [
    "faithfulness",
    "key_claim_recall",
    "topic_relevance",
    "methodological_completeness",
    "statistical_factuality",
    "contradiction_recognition",
]

METRIC_LABELS = [
    "Faithfulness",
    "Key Claim\nRecall",
    "Topic\nRelevance",
    "Methodological\nCompleteness",
    "Statistical\nFactuality",
    "Contradiction\nRecognition",
]

PALETTE = ["#4F86C6", "#F4A261", "#2A9D8F", "#E76F51", "#8ECAE6", "#A8DADC"]
STRICT_COLOR = "#E63946"
STANDARD_COLOR = "#457B9D"


def _short_topic(topic: str, max_len: int = 38) -> str:
    return topic if len(topic) <= max_len else topic[:max_len].rstrip() + "…"


def plot_revision_histogram(all_revisions: list[int], max_rev: int) -> plt.Figure:
    counts = [all_revisions.count(x) for x in range(max_rev + 1)]
    x_labels = [f"{x} Rev." for x in range(max_rev + 1)]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x_labels, counts, color=PALETTE[:len(counts)], edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                str(val), ha="center", fontsize=13, fontweight="bold")

    ax.set_title("Distribution of Revision Counts Across Topics", fontsize=14, pad=14, fontweight="bold")
    ax.set_ylabel("Number of Topics", fontsize=12)
    ax.set_xlabel("Revisions Needed to Pass the Judge", fontsize=12)
    ax.set_ylim(0, max(counts) + 2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_metrics_heatmap(all_scores: list[dict], topics: list[str]) -> plt.Figure:
    data = np.array([[row.get(m, 0.0) for m in METRIC_NAMES] for row in all_scores])
    short_topics = [_short_topic(t) for t in topics]

    fig_h = max(6, len(topics) * 0.35)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    cmap = LinearSegmentedColormap.from_list("rg", ["#E63946", "#F4A261", "#2A9D8F"])
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(METRIC_NAMES)))
    ax.set_xticklabels(METRIC_LABELS, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(short_topics)))
    ax.set_yticklabels(short_topics, fontsize=8)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Score", fraction=0.02, pad=0.02)
    ax.set_title("Evaluation Metrics Heatmap (per Topic)", fontsize=13, pad=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_metrics_summary(all_scores: list[dict]) -> plt.Figure:
    labels = [lbl.replace("\n", " ") for lbl in METRIC_LABELS]
    means = [np.mean([s[m] for s in all_scores]) for m in METRIC_NAMES]
    mins = [np.min([s[m] for s in all_scores]) for m in METRIC_NAMES]
    maxs = [np.max([s[m] for s in all_scores]) for m in METRIC_NAMES]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 7))
    rects1 = ax.bar(x - width, mins, width, label='Min', color='#E63946')
    rects2 = ax.bar(x, means, width, label='Mean', color='#457B9D')
    rects3 = ax.bar(x + width, maxs, width, label='Max', color='#2A9D8F')

    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Metric Summary (Min, Mean, Max)', fontweight="bold", fontsize=15, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    fig.tight_layout()
    return fig


def run_evaluation():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("run.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    agent = create_agent_graph()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Bachelor_Thesis")

    all_revisions: list[int] = []
    all_scores: list[dict] = []
    topics_run: list[str] = []

    total_dataset_tokens_cheap = 0
    total_dataset_tokens_smart = 0
    total_dataset_tokens_judge = 0

    logger.info(f"Starting evaluation of {len(DATASET)} topics...")

    with mlflow.start_run(run_name="Dataset_Evaluation_V1") as parent_run:

        mlflow.log_param("total_topics", len(DATASET))
        mlflow.log_param("max_revisions_allowed", CONFIG["evaluation"]["max_revisions"])
        mlflow.log_param("cheap_model", CONFIG["llm"]["cheap_model"])
        mlflow.log_param("smart_model", CONFIG["llm"]["smart_model"])
        mlflow.log_param("judge_model", CONFIG["llm"]["judge_model"])

        for i, topic in enumerate(DATASET):
            logger.info(f"\n[{i + 1}/{len(DATASET)}] Testing: {topic}")
            topics_run.append(topic)

            run_title = f"Topic_{i + 1}: {_short_topic(topic, 60)}"
            with mlflow.start_run(run_name=run_title, nested=True):
                initial_state = {
                    "topic": topic, "perspectives": [], "search_queries": [],
                    "raw_papers": [], "filtered_papers": [], "draft_report": "",
                    "final_report": "", "evaluation_feedback": "", "revision_count": 0,
                    "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "tokens_cheap": 0, "tokens_smart": 0, "tokens_judge": 0,
                }

                mlflow.log_param("topic", topic)
                result = agent.invoke(initial_state)

                revisions_taken = result.get("revision_count", 0)
                mlflow.log_metric("revisions", revisions_taken)
                all_revisions.append(revisions_taken)

                mlflow.log_metric("total_tokens", result.get("total_tokens", 0))
                mlflow.log_metric("prompt_tokens", result.get("prompt_tokens", 0))
                mlflow.log_metric("completion_tokens", result.get("completion_tokens", 0))

                cheap_tk = result.get("tokens_cheap", 0)
                smart_tk = result.get("tokens_smart", 0)
                judge_tk = result.get("tokens_judge", 0)
                mlflow.log_metric("tokens_cheap", cheap_tk)
                mlflow.log_metric("tokens_smart", smart_tk)
                mlflow.log_metric("tokens_judge", judge_tk)

                total_dataset_tokens_cheap += cheap_tk
                total_dataset_tokens_smart += smart_tk
                total_dataset_tokens_judge += judge_tk

                if revisions_taken > 0 and result.get("evaluation_feedback"):
                    mlflow.log_text(result["evaluation_feedback"], "revision_feedback.txt")

                if result.get("draft_report"):
                    mlflow.log_text(result["draft_report"], "draft_report.txt")

                final_draft = result.get("final_report", "") or result.get("draft_report", "")
                if final_draft:
                    mlflow.log_text(final_draft, "final_report.txt")

                sources = str(result.get("filtered_papers", ""))
                if final_draft and sources:
                    eval_result, _ = evaluate_all_metrics_super_judge(
                        llm_judge, sources, final_draft, topic
                    )
                    final_metric_scores = {
                        m: eval_result.get(m, {}).get("score", 0.0)
                        for m in METRIC_NAMES
                    }
                    all_scores.append(final_metric_scores)
                else:
                    all_scores.append({m: 0.0 for m in METRIC_NAMES})

        logger.info("\nEvaluation finished! Generating statistics and plots...")

        avg_revisions = sum(all_revisions) / len(all_revisions) if all_revisions else 0
        mlflow.log_metric("average_revisions_per_topic", avg_revisions)

        mlflow.log_metric("total_dataset_tokens_cheap", total_dataset_tokens_cheap)
        mlflow.log_metric("total_dataset_tokens_smart", total_dataset_tokens_smart)
        mlflow.log_metric("total_dataset_tokens_judge", total_dataset_tokens_judge)
        mlflow.log_metric("total_dataset_tokens_all",
                          total_dataset_tokens_cheap + total_dataset_tokens_smart + total_dataset_tokens_judge)

        if all_scores:
            avg_scores = {
                m: float(np.mean([s.get(m, 0.0) for s in all_scores]))
                for m in METRIC_NAMES
            }
            for m, v in avg_scores.items():
                mlflow.log_metric(f"avg_{m}", v)

        fig1 = plot_revision_histogram(all_revisions, CONFIG["evaluation"]["max_revisions"])
        mlflow.log_figure(fig1, "plots/01_revision_histogram.png")
        plt.close(fig1)

        if all_scores:
            fig2 = plot_metrics_heatmap(all_scores, topics_run)
            mlflow.log_figure(fig2, "plots/02_metrics_heatmap.png")
            plt.close(fig2)

            fig3 = plot_metrics_summary(all_scores)
            mlflow.log_figure(fig3, "plots/03_metrics_summary.png")
            plt.close(fig3)

    print(f"\nEvaluation finished! Results logged to MLflow parent run: {parent_run.info.run_id}")


if __name__ == "__main__":
    run_evaluation()
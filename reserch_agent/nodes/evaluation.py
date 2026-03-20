import mlflow
import logging
from ..llm import llm_smart
from ..models import AgentState
from ..config import  CONFIG
from ..metrics.evaluators import evaluate_all_metrics_super_judge

logger = logging.getLogger(__name__)

def evaluator_node(state: AgentState):
    revision = state.get("revision_count", 0)
    logger.info(f"\n   Start of LLM-Judge (Loop {revision + 1})...")

    draft = state.get("draft_report", "")
    sources = str(state.get("filtered_papers", ""))
    topic = state.get("topic", "")

    strict_thr = CONFIG["evaluation"]["strict_threshold"]
    std_thr = CONFIG["evaluation"]["standard_threshold"]

    results = evaluate_all_metrics_super_judge(llm_smart, sources, draft, topic)

    feedback = []
    for metric_name, data in results.items():
        score = data["score"]
        error_text = data["errors"]
        logger.debug(f'  Metric: {metric_name}  |  Score: {score}\n   Response: {error_text}')
        mlflow.log_metric(metric_name, score, step=revision)

        threshold = strict_thr if metric_name in ["faithfulness", "statistical_factuality"] else std_thr
        if score < threshold:
            feedback.append(f"- {metric_name.replace('_', ' ').title()}: {error_text}")

    if feedback:
        final_feedback = "Errors:\n" + "\n".join(feedback)

        next_revision = revision + 1
        if next_revision >= CONFIG["evaluation"]["max_revisions"]:
            logger.warning(" Max revisions reached! Saving the current draft as Final Report despite errors.")
            return {
                "evaluation_feedback": final_feedback,
                "revision_count": next_revision,
                "final_report": draft
            }

        logger.info("Report failed some metrics. Sending to Reviser...")
        return {
            "evaluation_feedback": final_feedback,
            "revision_count": next_revision
        }
    else:
        logger.info(" Report passed all tests!")
        return {
            "evaluation_feedback": "",
            "revision_count": revision + 1,
            "final_report": draft
        }


def check_quality(state: AgentState):
    feedback = state.get("evaluation_feedback", "")
    revisions = state.get("revision_count", 0)


    if revisions >= CONFIG["evaluation"]["max_revisions"]:
        logger.info("🛑 Limit of tries is reached. Final answer:")
        return "end_process"

    if feedback == "":
        return "end_process"
    else:
        return "needs_revision"

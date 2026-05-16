import sys
import matplotlib
matplotlib.use("Agg")
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("run.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

from reserch_agent.graph import create_agent_graph

logger = logging.getLogger(__name__)


def run_single_topic(topic: str) -> str:
    """Run the research agent on a single topic and return the final report."""
    agent = create_agent_graph()

    initial_state = {
        "topic": topic,
        "perspectives": [],
        "search_queries": [],
        "raw_papers": [],
        "filtered_papers": [],
        "draft_report": "",
        "final_report": "",
        "evaluation_feedback": "",
        "revision_count": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tokens_cheap": 0,
        "tokens_smart": 0,
        "tokens_judge": 0,
    }

    logger.info(f"Starting research on: {topic}")
    result = agent.invoke(initial_state)

    final_report = result.get("final_report", "") or result.get("draft_report", "")
    revisions = result.get("revision_count", 0)
    total_tokens = result.get("total_tokens", 0)

    logger.info(f"Research complete. Revisions: {revisions}, Tokens used: {total_tokens}")
    return final_report


if __name__ == "__main__":
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = input("Enter research topic: ").strip()
        if not topic:
            print("No topic provided. Exiting.")
            sys.exit(1)

    report = run_single_topic(topic)

    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(report)

    with open("final_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to final_report.txt")
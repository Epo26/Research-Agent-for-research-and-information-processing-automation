import mlflow
import logging
from reserch_agent.graph import create_agent_graph
from reserch_agent.config import PROMPTS, CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

agent = create_agent_graph()

if __name__ == "__main__":

    user_input = input("Enter academic topic: ")

    logger.info(f"Starting research pipeline for topic: '{user_input}'")

    initial_state = {
        "topic": user_input,
        "perspectives": [],
        "search_queries": [],
        "raw_papers": [],
        "filtered_papers": [],
        "draft_report": "",
        "final_report": "",
        "evaluation_feedback": "",
        "revision_count": 0
    }

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Bachelor_Research_Agent")



    with mlflow.start_run(run_name=f"{user_input}"):
        logger.info("MLflow run started.")
        mlflow.log_param("topic", user_input)
        mlflow.log_dict(CONFIG, "config.yaml")
        mlflow.log_dict(PROMPTS, "prompts.yaml")

        logger.info("Invoking LangGraph agent...")

        result = agent.invoke(initial_state)

        logger.info("Agent pipeline finished successfully!")

        with open("final_report.txt", "w", encoding="utf-8") as f:
            f.write(result["final_report"])
        mlflow.log_artifact("final_report.txt")


    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50)
    print(result["final_report"])
    print("\n" + "=" * 50)
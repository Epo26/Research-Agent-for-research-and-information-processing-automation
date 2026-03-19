from graph import create_agent_graph
from tools_and_nodes import llm_smart, llm_judge
import mlflow



agent = create_agent_graph()

if __name__ == "__main__":

    user_input = input("Enter academic topic: ")

    initial_state = {
        "topic": user_input,
        "search_queries": [],
        "raw_papers": [],
        "filtered_papers": [],
        "final_report": ""
    }

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    mlflow.set_experiment("Bachelor_Research_Agent")


    with mlflow.start_run(run_name=f"{user_input}"):
        print("✅ MLflow server!")
        mlflow.log_param("model", llm_smart.model_name)
        mlflow.log_param("temperature", llm_smart.temperature)
        mlflow.log_param("topic", user_input)

        result = agent.invoke(initial_state)

        source_abstracts = "\n\n".join(
            [f"Title: {p['title']}\nAbstract: {p['summary']}" for p in result.get("filtered_papers", [])]
        )

        retrieved_papers_count = len(result.get("filtered_papers", []))

        print("[Metrics] Calculating evaluation metrics...")


        with open("final_report.txt", "w", encoding="utf-8") as f:
            f.write(result["final_report"])
        mlflow.log_artifact("final_report.txt")


    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50)
    if result["final_report"] == "":
        print(result["draft_report"])
    else:
        print(result["final_report"])
    print("\n" + "=" * 50)
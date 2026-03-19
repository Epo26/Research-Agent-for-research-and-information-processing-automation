import arxiv
import os
import mlflow
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict, Dict
from langchain_groq import ChatGroq
from metrics import *
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.retrievers import PubMedRetriever
from pypdf import PdfReader


with open("config.yaml", "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

with open("prompts.yaml", "r", encoding="utf-8") as file:
    PROMPTS = yaml.safe_load(file)


load_dotenv()

arxiv_wrapper = ArxivAPIWrapper(top_k_results=2)

llm_cheap = llm_smart = ChatGroq(
    model= CONFIG["llm"]["cheap_model"],
    temperature= CONFIG["llm"]["cheap_temperature"]
)

llm_smart = ChatGroq(
    model= CONFIG["llm"]["smart_model"],
    temperature= CONFIG["llm"]["smart_temperature"]
)

llm_judge = ChatGroq(
    model= CONFIG["llm"]["judge_model"],
    temperature= CONFIG["llm"]["judge_temperature"]
)

class AgentState(TypedDict):
    topic: str
    perspectives: List[str]
    search_queries: List[str]
    raw_papers: List[Dict]
    filtered_papers: List[Dict]
    draft_report: str
    final_report: str
    evaluation_feedback: str
    revision_count: int


def perspective_generation_node(state: AgentState):

    print("Generating research perspectives...")
    topic = state["topic"]

    prompt_template = PROMPTS["nodes_prompts"]["perspective_generation_prompt"]
    prompt = prompt_template.format(topic=topic)


    response = llm_cheap.invoke(prompt)

    limit = CONFIG["search"]["max_perspectives"]
    perspectives = [p.strip() for p in response.content.split('\n') if p.strip()][:limit]

    print(f"   Identified Perspectives:\n" + "\n".join([f"   - {p}" for p in perspectives]))
    return {"perspectives": perspectives}



def query_expansion_node(state: AgentState):
    print("🔄 Expanding search queries for all perspectives (Batch Mode)...")
    topic = state.get("topic", "")
    perspectives = state.get("perspectives", [])

    perspectives_text = "\n".join([f"- {p}" for p in perspectives])

    prompt_template = PROMPTS["nodes_prompts"]["query_expansion_prompt"]
    prompt = prompt_template.format(topic=topic,perspectives_text=perspectives_text)


    try:
        response = llm_cheap.invoke(prompt)
        raw_text = response.content.strip()

        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        clean_json = json_match.group(0) if json_match else raw_text

        queries = json.loads(clean_json)

        for p, q in zip(perspectives, queries):
            print(f"   ✓ Query for [{p[:30]}...]: {q}")

        return {"search_queries": queries}

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON error in Query Expansion: {e}")
        fallback_queries = [" ".join(p.split()[:3]) for p in perspectives]
        return {"search_queries": fallback_queries}
    except Exception as e:
        print(f"⚠️ System Error: {e}")
        return {"search_queries": []}


def search_arxiv(state: AgentState):
    print("🌐 Searching ArXiv for all queries...")
    queries = state["search_queries"]
    all_papers = []
    seen_ids = set()


    for query in queries:
        print(f"   -> Executing: {query}")
        try:
            search = arxiv.Search(query=query, max_results= CONFIG["search"]["max_arxiv_results"])
            for result in search.results():
                if result.entry_id not in seen_ids:
                    seen_ids.add(result.entry_id)
                    all_papers.append({
                        "id": result.entry_id,
                        "title": result.title,
                        "summary": result.summary.replace('\n', ' '),
                        "authors": [a.name for a in result.authors]
                    })
        except Exception as e:
            print(f"   ⚠️ Search failed for '{query}': {e}")

    print(f"   Total unique papers fetched: {len(all_papers)}")
    return {"raw_papers": all_papers}


def relevance_filter_node(state: AgentState):

    print("Filtering relevant papers...")
    topic = state["topic"]
    raw_papers = state["raw_papers"]
    filtered = []
    crop_len = CONFIG["search"]["abstract_crop_length"]

    if not raw_papers:
        return {"filtered_papers": []}

    for paper in raw_papers:


        prompt_template = PROMPTS["nodes_prompts"]["relevance_filter_prompt"]
        prompt = prompt_template.format(topic=topic, paper_title=paper["title"], paper_summary=paper['summary'][:crop_len])

        response = llm_cheap.invoke(prompt)

        if "YES" in response.content.upper():
            filtered.append(paper)
            print(f"   ✅ Kept: {paper['title'][:50]}...")
        else:
            print(f"   ❌ Discarded: {paper['title'][:50]}...")

    return {"filtered_papers": filtered}



def synthesis_node(state: AgentState):

    print("Performing Citation-Aware Synthesis...")
    topic = state["topic"]
    papers = state["filtered_papers"]

    if not papers:
        return {"final_report": "Can't find enough papers for research."}


    context = ""
    references_list = "\n\n### References:\n"

    for i, p in enumerate(papers, 1):
        context += f"--- SOURCE [{i}] ---\nTitle: {p['title']}\nAuthors: {', '.join(p['authors'])}\nAbstract: {p['summary']}\n\n"
        references_list += f"[{i}] {', '.join(p['authors'])}. \"{p['title']}\".\n"

    prompt_template = PROMPTS["nodes_prompts"]["synthesis_prompt"]
    prompt = prompt_template.format(topic=topic, context=context)

    response = llm_smart.invoke(prompt)

    report_text = response.content + references_list

    return {"draft_report": report_text}



def evaluator_node(state: AgentState):
    revision = state.get("revision_count", 0)
    print(f"\n⚖️ Start of LLM-Judge (Loop {revision + 1})...")

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
        print(f'  Metric: {metric_name}  |  Score: {score}\n   Response: {error_text}')
        mlflow.log_metric(metric_name, score, step=revision)

        threshold = strict_thr if metric_name in ["faithfulness", "statistical_factuality"] else std_thr
        if score < threshold:
            feedback.append(f"- {metric_name.replace('_', ' ').title()}: {error_text}")

    if feedback:
        final_feedback = "Errors:\n" + "\n".join(feedback)
        return {"evaluation_feedback": final_feedback, "revision_count": revision + 1}
    else:
        print("   ✅ Report passed all tests!")
        return {
            "evaluation_feedback": "",
            "revision_count": revision + 1,
            "final_report": draft
        }



def reviser_node(state: AgentState):
    print(f"🔄 New try (Attempt: {state['revision_count']})...")

    draft = state.get("draft_report", "Draft report is not found.")
    feedback = state.get("evaluation_feedback", "No feedback.")
    sources = str(state.get("filtered_papers", ""))

    prompt_template = PROMPTS["nodes_prompts"]["reviser_prompt"]
    prompt = prompt_template.format(sources=sources, draft=draft,feedback=feedback)

    response = llm_smart.invoke([HumanMessage(content=prompt)])

    return {"draft_report": response.content}

def check_quality(state: AgentState):
    feedback = state.get("evaluation_feedback", "")
    revisions = state.get("revision_count", 0)


    if revisions >= CONFIG["evaluation"]["max_revisions"]:
        print("🛑 Limit of tries is reached. Final answer:")
        return "end_process"

    if feedback == "":
        return "end_process"
    else:
        return "needs_revision"
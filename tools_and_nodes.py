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


load_dotenv()

arxiv_wrapper = ArxivAPIWrapper(top_k_results=2)

llm_cheap = llm_smart = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

llm_smart = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1
)

llm_judge = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1
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

    prompt = f"""
    You are an expert academic editor planning a comprehensive, Wikipedia-style article on the topic: '{topic}'.
    Identify from 5 essential and distinct perspectives (or sub-topics) that MUST be researched to provide a complete overview.
    For example, if the topic is 'Large Language Models', perspectives might be 'Core Architecture', 'Ethical Implications', and 'Real-world Applications'.

    Return ONLY 5 of the perspectives, one per line. Do not use bullet points or extra text.
    """
    response = llm_cheap.invoke(prompt)
    perspectives = [p.strip() for p in response.content.split('\n') if p.strip()][:5]

    print(f"   Identified Perspectives:\n" + "\n".join([f"   - {p}" for p in perspectives]))
    return {"perspectives": perspectives}


import json
import re


def query_expansion_node(state: AgentState):
    print("🔄 Expanding search queries for all perspectives (Batch Mode)...")
    topic = state.get("topic", "")
    perspectives = state.get("perspectives", [])

    perspectives_text = "\n".join([f"- {p}" for p in perspectives])

    prompt = f"""
    You are an expert academic librarian. Your task is to generate search queries for the topic: {topic}.

    Here are the specific perspectives you need to cover:
    {perspectives_text}

    CRITICAL RULES FOR SEARCH QUERIES:
    1. Generate exactly ONE search query for EACH perspective provided above.
    2. Keep queries EXTREMELY SIMPLE (maximum 3-4 keywords).
    3. DO NOT use boolean operators (AND, OR, NOT).
    4. DO NOT use parentheses (). 
    5. ArXiv's API will crash if the query is too complex. 

    Good example query: "LoRA memory efficiency LLM"
    Bad example query: "(LoRA OR QLoRA) AND (memory) AND (LLM)"

    Return ONLY a valid JSON list of strings. Do not include markdown formatting like ```json.
    Format example:
    [
      "keyword1 keyword2 keyword3",
      "keyword4 keyword5 keyword6"
    ]
    """

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
            search = arxiv.Search(query=query, max_results=2)
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

    if not raw_papers:
        return {"filtered_papers": []}

    for paper in raw_papers:

        prompt = f"""
        Topic: {topic}
        Paper Title: {paper['title']}
        Paper Abstract: {paper['summary'][:1000]}...

        Is this paper highly relevant to the topic? 
        Answer ONLY 'YES' or 'NO'.
        """
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

    prompt = f"""
    You are an expert academic writer and researcher. Your task is to write a comprehensive academic summary on the topic: '{topic}'.

    CRITICAL RULES:
    1. You MUST use ONLY the information provided in the sources below. Do not use any external knowledge.
    2. Every factual claim, sentence, or methodology you mention MUST be immediately followed by an inline citation using the source ID, like [1] or [2, 3].
    3. If the sources do not contain enough information to cover a sub-topic, explicitly state "The provided literature does not cover this aspect." Do not invent information.
    4. Maintain a formal, objective, and academic tone.

    AVAILABLE SOURCES:
    {context}

    Structure the report with the following sections:
    - Introduction
    - Key Findings & Methodologies
    - Limitations & Future Directions
    """

    response = llm_smart.invoke(prompt)

    report_text = response.content + references_list

    return {"draft_report": report_text}



def evaluator_node(state: AgentState):
    revision = state.get("revision_count", 0)
    print(f"\n⚖️ Start of LLM-Judge (Loop {revision + 1})...")

    draft = state.get("draft_report", "")
    sources = str(state.get("filtered_papers", ""))
    topic = state.get("topic", "")

    results = evaluate_all_metrics_super_judge(llm_smart, sources, draft, topic)

    feedback = []
    for metric_name, data in results.items():
        score = data["score"]
        error_text = data["errors"]
        print(f'  Metric: {metric_name}  |  Score: {score}\n   Response: {error_text}')
        mlflow.log_metric(metric_name, score, step=revision)

        threshold = 0.9 if metric_name in ["faithfulness", "statistical_factuality"] else 0.8
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

    prompt = f"""
    You are an expert academic writer. You wrote a draft report, but the reviewers (judges) found mistakes.

    Source Documents:
    {sources}

    Your Draft Report:
    {draft}

    Reviewer Feedback / Errors to fix:
    {feedback}

    Task: Rewrite the draft report to fix ALL the errors mentioned by the reviewers. 
    Ensure you maintain an academic tone and accurately cite the source documents.
    """

    response = llm_smart.invoke([HumanMessage(content=prompt)])

    return {"draft_report": response.content}

def check_quality(state: AgentState):
    feedback = state.get("evaluation_feedback", "")
    revisions = state.get("revision_count", 0)

    if revisions >= 3:
        print("🛑 Limit of tries is reached. Final answer:")
        return "end_process"

    if feedback == "":
        return "end_process"
    else:
        return "needs_revision"
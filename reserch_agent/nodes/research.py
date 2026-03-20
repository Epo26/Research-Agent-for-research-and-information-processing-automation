import re
import json
import arxiv
import logging
import concurrent.futures
from ..llm import llm_cheap
from ..models import AgentState, Paper
from ..config import PROMPTS, CONFIG

logger = logging.getLogger(__name__)

def perspective_generation_node(state: AgentState):

    logger.info("Generating research perspectives...")
    topic = state["topic"]

    prompt_template = PROMPTS["nodes_prompts"]["perspective_generation_prompt"]
    prompt = prompt_template.format(topic=topic)


    response = llm_cheap.invoke(prompt)

    limit = CONFIG["search"]["max_perspectives"]
    perspectives = [p.strip() for p in response.content.split('\n') if p.strip()][:limit]
    logger.debug(f"   Identified Perspectives:\n" + "\n".join([f"   - {p}" for p in perspectives]))
    return {"perspectives": perspectives}


def query_expansion_node(state: AgentState):
    logger.info("Expanding search queries for all perspectives...")
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
            logger.debug(f" ✓ Query for [{p}]: {q}")

        return {"search_queries": queries}

    except json.JSONDecodeError as e:
        logger.warning(f"JSON error in Query Expansion: {e}")
        fallback_queries = [" ".join(p.split()[:3]) for p in perspectives]
        return {"search_queries": fallback_queries}
    except Exception as e:
        logger.warning(f"System Error: {e}")
        return {"search_queries": []}


def search_arxiv(state: AgentState):
    logger.info("Searching ArXiv for all queries (Parallel Mode)...")
    queries = state.get("search_queries", [])

    if not queries:
        logger.warning("No search queries provided to ArXiv node.")
        return {"raw_papers": []}

    all_papers = []
    seen_ids = set()
    max_results = CONFIG["search"]["max_arxiv_results"]

    def fetch_single_query(query: str):
        logger.debug(f"   -> Executing: {query}")
        papers_found = []
        try:
            search = arxiv.Search(query=query, max_results=max_results)
            for result in search.results():
                papers_found.append(
                    Paper(
                        id=result.entry_id,
                        title=result.title,
                        summary=result.summary.replace('\n', ' '),
                        authors=[a.name for a in result.authors]
                    )
                )
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")

        return papers_found

    max_workers = min(10, len(queries))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {executor.submit(fetch_single_query, q): q for q in queries}

        for future in concurrent.futures.as_completed(future_to_query):
            papers = future.result()

            for paper in papers:
                if paper.id not in seen_ids:
                    seen_ids.add(paper.id)
                    all_papers.append(paper)

    logger.info(f"Total unique papers fetched: {len(all_papers)}")
    return {"raw_papers": all_papers}

def relevance_filter_node(state: AgentState):

    logger.info("Filtering relevant papers...")
    topic = state["topic"]
    raw_papers = state["raw_papers"]
    filtered = []
    crop_len = CONFIG["search"]["abstract_crop_length"]

    if not raw_papers:
        return {"filtered_papers": []}

    for paper in raw_papers:


        prompt_template = PROMPTS["nodes_prompts"]["relevance_filter_prompt"]
        prompt = prompt_template.format(topic=topic, paper_title=paper.title, paper_summary=paper.summary[:crop_len])

        response = llm_cheap.invoke(prompt)

        if "YES" in response.content.upper():
            filtered.append(paper)
            logger.debug(f"   ✅ Kept: {paper.title[:50]}...")
        else:
            logger.debug(f"   ❌ Discarded: {paper.title[:50]}...")

    return {"filtered_papers": filtered}
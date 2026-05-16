import re
import json
import time
import arxiv
import logging
from ..llm import llm_cheap
from ..models import AgentState, Paper
from ..config import PROMPTS, CONFIG
from ..utils import update_token_usage, safe_invoke

logger = logging.getLogger(__name__)

def perspective_generation_node(state: AgentState):

    logger.info("Generating research perspectives...")
    topic = state["topic"]

    prompt_template = PROMPTS["nodes_prompts"]["perspective_generation_prompt"]
    prompt = prompt_template.format(topic=topic)


    response = safe_invoke(llm_cheap, prompt)

    limit = CONFIG["search"]["max_perspectives"]
    perspectives = [p.strip() for p in response.content.split('\n') if p.strip()][:limit]
    logger.debug(f"   Identified Perspectives:\n" + "\n".join([f"   - {p}" for p in perspectives]))
    
    token_updates = update_token_usage(state, response, "cheap")
    return {"perspectives": perspectives, **token_updates}


def query_expansion_node(state: AgentState):
    logger.info("Expanding search queries for all perspectives...")
    topic = state.get("topic", "")
    perspectives = state.get("perspectives", [])

    perspectives_text = "\n".join([f"- {p}" for p in perspectives])

    prompt_template = PROMPTS["nodes_prompts"]["query_expansion_prompt"]
    prompt = prompt_template.format(topic=topic,perspectives_text=perspectives_text)


    try:
        response = safe_invoke(llm_cheap, prompt)
        raw_text = response.content.strip()

        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        clean_json = json_match.group(0) if json_match else raw_text

        queries = json.loads(clean_json)

        for p, q in zip(perspectives, queries):
            logger.debug(f"  Query for [{p}]: {q}")

        token_updates = update_token_usage(state, response, "cheap")
        return {"search_queries": queries, **token_updates}

    except json.JSONDecodeError as e:
        logger.warning(f"JSON error in Query Expansion: {e}")
        fallback_queries = [" ".join(p.split()[:3]) for p in perspectives]
        return {"search_queries": fallback_queries}
    except Exception as e:
        logger.warning(f"System Error: {e}")
        return {"search_queries": []}


def search_arxiv(state: AgentState):
    logger.info("Searching ArXiv for all queries (Sequential Mode, throttled)...")
    queries = state.get("search_queries", [])

    if not queries:
        logger.warning("No search queries provided to ArXiv node.")
        return {"raw_papers": []}

    all_papers = []
    seen_ids = set()
    max_results = CONFIG["search"]["max_arxiv_results"]
    # ArXiv public API allows ~1 request per 3 s per IP.
    # Sequential + delay avoids HTTP 429 errors that return 0 papers.
    throttle_delay = CONFIG["search"].get("arxiv_throttle_delay", 4)

    for query in queries:
        logger.debug(f"   -> Executing: {query}")
        try:
            search = arxiv.Search(query=query, max_results=max_results)
            for result in search.results():
                if result.entry_id not in seen_ids:
                    seen_ids.add(result.entry_id)
                    all_papers.append(
                        Paper(
                            id=result.entry_id,
                            title=result.title,
                            summary=result.summary.replace('\n', ' '),
                            authors=[a.name for a in result.authors]
                        )
                    )
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")

        time.sleep(throttle_delay)

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

    token_updates = {"total_tokens": state.get("total_tokens", 0), 
                     "prompt_tokens": state.get("prompt_tokens", 0), 
                     "completion_tokens": state.get("completion_tokens", 0),
                     "tokens_cheap": state.get("tokens_cheap", 0)}

    for paper in raw_papers:
        prompt_template = PROMPTS["nodes_prompts"]["relevance_filter_prompt"]
        prompt = prompt_template.format(topic=topic, paper_title=paper.title, paper_summary=paper.summary[:crop_len])

        response = safe_invoke(llm_cheap, prompt)
        token_updates = update_token_usage(token_updates, response, "cheap")

        if "YES" in response.content.upper():
            filtered.append(paper)
            logger.debug(f"   Kept: {paper.title[:50]}...")
        else:
            logger.debug(f"   Discarded: {paper.title[:50]}...")

        time.sleep(2.0)

    return {"filtered_papers": filtered, **token_updates}
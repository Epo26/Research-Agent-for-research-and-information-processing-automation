import logging
from langchain_core.messages import HumanMessage
from ..llm import llm_smart
from ..models import AgentState, Paper
from ..config import PROMPTS, CONFIG
from ..utils import update_token_usage, safe_invoke

logger = logging.getLogger(__name__)

def synthesis_node(state: AgentState):

    logger.info("Performing Citation-Aware Synthesis...")
    topic = state["topic"]
    papers = state["filtered_papers"]

    if not papers:
        return {"final_report": "Can't find enough papers for research."}


    context = ""
    references_list = "\n\n### References:\n"

    for i, p in enumerate(papers, 1):
        context += f"--- SOURCE [{i}] ---\nTitle: {p.title}\nAuthors: {', '.join(p.authors)}\nAbstract: {p.summary}\n\n"
        references_list += f"[{i}] {', '.join(p.authors)}. \"{p.title}\".\n"

    prompt_template = PROMPTS["nodes_prompts"]["synthesis_prompt"]
    prompt = prompt_template.format(topic=topic, context=context)

    response = safe_invoke(llm_smart, prompt)

    report_text = response.content + references_list

    token_updates = update_token_usage(state, response, "smart")
    return {"draft_report": report_text, **token_updates}


def reviser_node(state: AgentState):
    logger.info(f"New try (Attempt: {state['revision_count']})...")

    draft = state.get("draft_report", "Draft report is not found.")
    feedback = state.get("evaluation_feedback", "No feedback.")
    sources = str(state.get("filtered_papers", ""))

    prompt_template = PROMPTS["nodes_prompts"]["reviser_prompt"]
    prompt = prompt_template.format(sources=sources, draft=draft,feedback=feedback)

    response = safe_invoke(llm_smart, [HumanMessage(content=prompt)])

    token_updates = update_token_usage(state, response, "smart")
    return {"draft_report": response.content, **token_updates}

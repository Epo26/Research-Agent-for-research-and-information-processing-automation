from langgraph.graph import StateGraph, START, END
from .models import AgentState
from .nodes.research import perspective_generation_node, query_expansion_node, search_arxiv, relevance_filter_node
from .nodes.synthesis import synthesis_node, reviser_node
from .nodes.evaluation import evaluator_node, check_quality



def create_agent_graph():
    builder = StateGraph(AgentState)

    builder.add_node("perspective_generation", perspective_generation_node)
    builder.add_node("query_expansion", query_expansion_node)
    builder.add_node("researcher", search_arxiv)
    builder.add_node("relevance_filter", relevance_filter_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("reviser", reviser_node)

    builder.add_edge(START, "perspective_generation")
    builder.add_edge("perspective_generation", "query_expansion")
    builder.add_edge("query_expansion", "researcher")
    builder.add_edge("researcher", "relevance_filter")
    builder.add_edge("relevance_filter", "synthesis")
    builder.add_edge("synthesis", "evaluator")

    builder.add_conditional_edges(
        "evaluator",
        check_quality,
        {
            "needs_revision": "reviser",
            "end_process": END
        }
    )

    builder.add_edge("reviser", "evaluator")


    return builder.compile()
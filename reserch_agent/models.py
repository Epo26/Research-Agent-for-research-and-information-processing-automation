import operator
from dataclasses import dataclass
from typing import Annotated, List, TypedDict, Dict

@dataclass
class Paper:
    id: str
    title: str
    summary: str
    authors: List[str]

class AgentState(TypedDict):
    topic: str
    perspectives: List[str]
    search_queries: List[str]
    raw_papers: Annotated[List[Paper], operator.add]
    filtered_papers: Annotated[List[Paper], operator.add]
    draft_report: str
    final_report: str
    evaluation_feedback: str
    revision_count: int

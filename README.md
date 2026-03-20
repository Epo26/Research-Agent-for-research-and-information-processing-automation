# Academic Research Agent (Bachelor's Thesis)

An automated AI-driven pipeline designed to conduct academic research, synthesize literature reviews, and rigorously evaluate the generated content using strict scientific metrics. Built with **LangGraph**, **LangChain**, and **MLflow**.

## Project Overview

This project automates the literature review process for academic research. Given a user topic, the agent dynamically generates research perspectives, queries the **ArXiv database**, filters relevant papers, synthesizes a citation-aware report, and uses an LLM-as-a-Judge mechanism (ReAct pattern) to self-correct hallucinations or factual inaccuracies.

### Key Features
* **Multi-Agent Workflow:** Orchestrated using LangGraph for stateful, cyclic execution.
* **Hybrid LLM Routing:** Uses cost-effective models (`llama-3.1-8b-instant`) for simple tasks and high-reasoning models (`llama-3.3-70b-versatile`) for synthesis and evaluation.
* **Scientific Super-Judge:** Evaluates drafts across 6 strict metrics:
  1. *Faithfulness (Hallucination Check)*
  2. *Key Claim Recall*
  3. *Topic Relevance*
  4. *Methodological Completeness*
  5. *Statistical Factuality*
  6. *Contradiction Recognition*
* **Self-Correction (ReAct):** Automatically rewrites the draft if quality thresholds (e.g., $<0.9$ for factuality) are not met.
* **MLOps Integration:** Fully integrated with **MLflow** to track metrics, parameters, and prompt versions for A/B testing and reproducibility.

---

## Project Architecture

```text
research_agent/
├── __init__.py
├── config.py             # Loads config.yaml & prompts.yaml
├── models.py             # Dataclasses (Paper) and TypedDict (AgentState)
├── llm.py                # LLM initializations (Groq API)
├── nodes/                # LangGraph Nodes
│   ├── research.py       # Perspective generation, Query expansion, ArXiv search
│   ├── synthesis.py      # Report drafting and Reviser logic
│   └── evaluation.py     # Super-Judge evaluation and conditional routing
├── metrics/
│   └── evaluators.py     # Complex scientific metric prompts and parsers
├── graph.py              # LangGraph compilation
├── prompts.yaml          # Externalized system prompts
├── config.yaml           # Hyperparameters (thresholds, max results, etc.)
└── main.py               # Entry point and MLflow tracking setup
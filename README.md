# Research Agent

An AI-driven pipeline that conducts academic research, synthesizes citation-aware literature reviews, and evaluates the output using strict scientific metrics. Built with **LangGraph**, **Google Gemini**, and **MLflow**.

## Quick start

### 1. Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- A Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### 2. Installation

```bash
git clone https://github.com/Epo26/Research-Agent-for-research-and-information-processing-automation.git
cd Research-Agent-for-research-and-information-processing-automation

# Create virtual environment and install dependencies
uv sync

# Or with pip
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -e .
```

### 3. Configuration

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Model parameters, search settings, and evaluation thresholds are configured in `config.yaml`.  
Prompts are externalized in `prompts.yaml`.

### 4. Run the agent on a single topic

```bash
python main.py "Mitigation strategies for hallucination in LLMs using knowledge graphs"
```

Or run interactively (you will be prompted to enter a topic):

```bash
python main.py
```

The generated report is printed to the console and saved to `final_report.txt`.

### 5. Run the full dataset evaluation (with MLflow tracking)

Start the MLflow server first:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then run the evaluation:

```bash
python run_dataset.py
```

This processes all 20 topics from `dataset.yaml`, logs metrics and artifacts to MLflow, and generates visualization plots (revision histogram, metrics heatmap, min/mean/max summary).

Open `http://127.0.0.1:5000` in a browser to explore the results.


### Evaluation metrics

The LLM Judge scores every report on 6 dimensions:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Faithfulness | 0.9 (strict) | No claims unsupported by sources |
| Statistical Factuality | 0.9 (strict) | Numerical data matches sources |
| Key Claim Recall | 0.8 | Important findings are covered |
| Topic Relevance | 0.8 | Report stays on-topic |
| Methodological Completeness | 0.8 | Methods from sources are described |
| Contradiction Recognition | 0.8 | Conflicting findings are noted |

If any metric falls below its threshold, the report is sent back to the Reviser for correction (up to 3 revision cycles).

### Hybrid LLM routing

The system uses two model tiers to balance cost and quality:

- **Gemini 2.5 Flash** (cheap) — perspective generation, query expansion, relevance filtering
- **Gemini 2.5 Pro** (smart) — report synthesis, revision, and evaluation

---

## Project structure

```
├── main.py                 # Single-topic entry point
├── run_dataset.py          # Batch evaluation with MLflow tracking and plots
├── config.yaml             # Model parameters and thresholds
├── prompts.yaml            # All LLM prompts
├── dataset.yaml            # 20 evaluation topics
├── pyproject.toml          # Dependencies
├── .env.example            # API key template
└── reserch_agent/
    ├── config.py            # YAML config loader
    ├── models.py            # Paper dataclass and AgentState TypedDict
    ├── llm.py               # LLM initialization (Google Gemini)
    ├── graph.py             # LangGraph state machine definition
    ├── utils.py             # Token tracking and fault-tolerant API wrapper
    ├── nodes/
    │   ├── research.py      # Perspective generation, query expansion, ArXiv search, filtering
    │   ├── synthesis.py     # Report synthesis and revision
    │   └── evaluation.py    # LLM Judge and quality gate routing
    └── metrics/
        └── evaluators.py    # Super-judge prompt and JSON response parser
```
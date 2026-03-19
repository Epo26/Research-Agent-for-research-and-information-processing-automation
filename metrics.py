import json
from langchain_core.messages import HumanMessage
import re
import yaml

with open("prompts.yaml", "r", encoding="utf-8") as file:
    PROMPTS = yaml.safe_load(file)

def evaluate_key_claim_recall(llm_judge, source_abstracts: str, generated_summary: str):

    prompt_template = PROMPTS["judges_prompts"]["evaluate_key_claim_recall"]
    eval_prompt = prompt_template.format(source_abstracts=source_abstracts, generated_summary=generated_summary)

    try:
        response = llm_judge.invoke([HumanMessage(content=eval_prompt)])

        raw_text = response.content.strip().strip('`').replace('json\n', '', 1)
        evaluation = json.loads(raw_text)

        score = int(evaluation.get("score", 0)) / 5

        print(f"   [Metric] Key Claim Recall: {score}")
        print(f"   [Metric Reason]: {evaluation.get('reasoning')}")

        return score, evaluation.get('reasoning')

    except Exception as e:
        print(f"⚠️ Error in metric key_claim_recall: {e}")
        return 0.0, []

def evaluate_topic_relevance(llm_judge, original_topic: str, generated_summary: str):

    prompt_template = PROMPTS["judges_prompts"]["evaluate_topic_relevance"]
    eval_prompt = prompt_template.format(original_topic=original_topic, generated_summary=generated_summary)

    try:
        response = llm_judge.invoke([HumanMessage(content=eval_prompt)])

        raw_text = response.content.strip().strip('`').replace('json\n', '', 1)
        evaluation = json.loads(raw_text)

        score = int(evaluation.get("score", 1)) / 10

        print(f"   [Metric] Topic Relevance: {score}")
        print(f"   [Metric Reason]: {evaluation.get('reasoning')}")

        return score, evaluation.get('reasoning')

    except Exception as e:
        print(f"⚠️ Error in metric Topic Relevance: {e}")
        return 0.0, []

def evaluate_comparative_coherence(llm_judge, source_abstracts: str, generated_summary: str):

    prompt_template = PROMPTS["judges_prompts"]["evaluate_comparative_coherence"]
    eval_prompt = prompt_template.format(source_abstracts=source_abstracts, generated_summary=generated_summary)

    try:
        response = llm_judge.invoke([HumanMessage(content=eval_prompt)])

        raw_text = response.content.strip().strip('`').replace('json\n', '', 1)
        evaluation = json.loads(raw_text)

        score = float(evaluation.get("score", 1.0))

        print(f"   [Metric] Comparative Coherence: {score}/1.0")
        print(f"   [Metric Reason]: {evaluation.get('reasoning')}")

        return score, evaluation.get('reasoning')

    except Exception as e:
        print(f"⚠️ Error in metric Comparative Coherence: {e}")
        return 0.0, []


def calculate_citation_density(final_report: str, retrieved_papers_count: int) -> float:

    if retrieved_papers_count == 0:
        return 0.0

    bracket_matches = re.findall(r'\[([\d,\s]+)\]', final_report)

    unique_citations = set()
    for match in bracket_matches:
        numbers = re.findall(r'\d+', match)
        for num in numbers:
            try:
                citation_idx = int(num)
                if 1 <= citation_idx <= retrieved_papers_count:
                    unique_citations.add(citation_idx)
            except ValueError:
                continue

    coverage_ratio = len(unique_citations) / retrieved_papers_count

    print(f"   [Metric] Citation Density: {len(unique_citations)}/{retrieved_papers_count} ({coverage_ratio:.2f})")
    return float(coverage_ratio)


def evaluate_faithfulness(llm_judge, source_abstracts: str, generated_summary: str):

    if not source_abstracts or not generated_summary:
        return 0.0, []

    prompt_template = PROMPTS["judges_prompts"]["evaluate_faithfulness"]
    eval_prompt = prompt_template.format(source_abstracts=source_abstracts, generated_summary=generated_summary)

    try:
        response = llm_judge.invoke([HumanMessage(content=eval_prompt)])

        raw_text = response.content.strip().strip('`').replace('json\n', '', 1)
        evaluation = json.loads(raw_text)

        score = float(evaluation.get("score", 1.0))
        hallucinations = evaluation.get("hallucinations", [])

        print(f"   [Metric] Faithfulness (Groundedness): {score:.2f}")
        if hallucinations and score < 1.0:
            print(f"   [Metric Hallucinations found]: {hallucinations}")

        return score, hallucinations

    except Exception as e:
        print(f"⚠️ Error in metric Faithfulness: {e}")
        return 0.0, []


def evaluate_methodological_completeness(llm_judge, source_abstracts: str, generated_summary: str):
    if not source_abstracts or not generated_summary:
        return 0.0, []

    prompt_template = PROMPTS["judges_prompts"]["evaluate_statistical_factuality"]
    eval_prompt = prompt_template.format(source_abstracts=source_abstracts, generated_summary=generated_summary)

    try:
        response = llm_judge.invoke([HumanMessage(content=eval_prompt)])

        raw_text = response.content.strip().strip('`').replace('json\n', '', 1)
        evaluation = json.loads(raw_text)

        score = float(evaluation.get("score", 0.0))
        missing = evaluation.get("missing_elements", [])

        print(f"   [Science Metric] Methodological Completeness: {score:.2f}")
        if missing and score < 1.0:
            print(f"   [Missing Details]: {missing}")

        return score, missing

    except Exception as e:
        print(f"⚠️ Error in metric Methodological Completeness: {e}")
        return 0.0, []


def evaluate_statistical_factuality(llm_judge, source_abstracts: str, generated_summary: str):

    if not source_abstracts or not generated_summary:
        return 0.0, ["Empty input"]

    prompt_template = PROMPTS["judges_prompts"]["evaluate_statistical_factuality"]
    eval_prompt = prompt_template.format(source_abstracts=source_abstracts, generated_summary=generated_summary)

    try:
        response = llm_judge.invoke([HumanMessage(content=eval_prompt)])
        raw_text = response.content.strip()

        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            clean_json_string = json_match.group(0)
        else:
            clean_json_string = raw_text

        evaluation = json.loads(clean_json_string)

        score = float(evaluation.get("score", 1.0))
        errors = evaluation.get("hallucinated_numbers", [])
        total = evaluation.get("total_claims", 0)
        found_errs = evaluation.get("errors_found", 0)

        print(f"   [Science Metric] Stat Factuality: {score:.2f} (Errors: {found_errs}/{total})")

        return score, errors

    except json.JSONDecodeError as e:
        print(f"⚠️ Error in JSON in Stat Factuality: {e}")
        return 0.5, ["Format error from evaluator. Please verify numbers carefully."]
    except Exception as e:
        print(f"⚠️ System error в Stat Factuality: {e}")
        return 0.0, [f"Evaluation Failed: {e}"]


def evaluate_contradiction_recognition(llm_judge, source_abstracts: str, generated_summary: str):

    if not source_abstracts or not generated_summary:
        return 0.0, []

    prompt_template = PROMPTS["judges_prompts"]["evaluate_contradiction_recognition"]
    eval_prompt = prompt_template.format(source_abstracts=source_abstracts, generated_summary=generated_summary)

    try:
        response = llm_judge.invoke([HumanMessage(content=eval_prompt)])

        raw_text = response.content.strip().strip('`').replace('json\n', '', 1)
        evaluation = json.loads(raw_text)

        score = float(evaluation.get("score", 1.0))
        explanation = evaluation.get("explanation", "No explanation provided.")

        print(f"   [Science Metric] Contradiction Recognition: {score:.2f}")
        if score < 1.0:
            print(f"   [Contradiction Handling]: {explanation}")

        return score, explanation

    except Exception as e:
        print(f"⚠️ Error in metric Contradiction Recognition: {e}")
        return 0.0 , []



def evaluate_all_metrics_super_judge(llm_judge, source_abstracts: str, generated_summary: str, topic: str):
    if not source_abstracts or not generated_summary:
        return {}

    prompt_template = PROMPTS["judges_prompts"]["evaluate_all_metrics_super_judge"]
    eval_prompt = prompt_template.format(topic=topic,source_abstracts=source_abstracts,generated_summary=generated_summary)


    try:
        response = llm_judge.invoke([HumanMessage(content=eval_prompt)])
        raw_text = response.content.strip()

        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        clean_json_string = json_match.group(0) if json_match else raw_text

        evaluation = json.loads(clean_json_string)
        return evaluation

    except Exception as e:
        print(f"⚠️ Error of judge: {e}")
        return {metric: {"score": 0.0, "errors": "Evaluation failed, please review draft."}
                for metric in ["faithfulness", "key_claim_recall", "topic_relevance", "methodological_completeness",
                               "statistical_factuality", "contradiction_recognition"]}
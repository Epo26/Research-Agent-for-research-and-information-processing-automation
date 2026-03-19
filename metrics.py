import json
from langchain_core.messages import HumanMessage
import re

def evaluate_key_claim_recall(llm_judge, source_abstracts: str, generated_summary: str):

    eval_prompt = f"""
    You are an expert academic evaluator. Your task is to objectively evaluate a research summary.

    Source Abstracts (Ground Truth):
    {source_abstracts}

    Generated Summary:
    {generated_summary}

    Task:
    1. Identify the 5 most significant scientific findings or claims present in the Source Abstracts.
    2. Critically analyze the Generated Summary.
    3. Count exactly how many of those 5 specific findings are explicitly present or accurately paraphrased in the Generated Summary.

    Return ONLY a valid JSON object. Do not include any other text, markdown formatting, or explanations outside the JSON.
    Strict Format:
    {{"score": <integer_between_0_and_5>, "reasoning": "<short_explanation_of_what_was_missed>"}}
    """

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

    eval_prompt = f"""
    You are an expert academic evaluator checking for "topic drift" in automated research pipelines.

    Original User Topic:
    "{original_topic}"

    Final Generated Summary:
    {generated_summary}

    Task:
    Evaluate how strictly the Final Generated Summary adheres to the Original User Topic. 
    Did the research pipeline drift into merely "related" but incorrect territories, or did it stay perfectly focused on the exact premise requested by the user?

    Score the relevance on a scale from 1 to 10, where:
    1-3: Severe topic drift. The summary discusses adjacent topics but misses the core premise entirely.
    4-7: Moderate relevance. Covers some core aspects but drifted into broader or unrelated sub-topics.
    8-10: Highly relevant. The summary stayed perfectly on-topic without unnecessary deviations.

    Return ONLY a valid JSON object. Do not include any other text or markdown formatting outside the JSON.
    Strict Format:
    {{"score": <integer_between_1_and_10>, "reasoning": "<short_explanation_of_drift_or_focus>"}}
    """

    try:
        response = llm_judge.invoke([HumanMessage(content=eval_prompt)])

        raw_text = response.content.strip().strip('`').replace('json\n', '', 1)
        evaluation = json.loads(raw_text)

        score = int(evaluation.get("score", 1)) / 10

        print(f"   [Metric] Topic Relevance: {score}")
        print(f"   [Metric Reason]: {evaluation.get('reasoning')}")

        return score, evaluation.get('reasoning')

    except Exception as e:
        print(f"⚠⚠️ Error in metric Topic Relevance: {e}")
        return 0.0, []

def evaluate_comparative_coherence(llm_judge, source_abstracts: str, generated_summary: str):

    eval_prompt = f"""
    You are an expert academic evaluator focusing on how well literature reviews handle conflicting scientific information.

    Source Abstracts:
    {source_abstracts}

    Generated Summary:
    {generated_summary}

    Task:
    1. First, analyze the Source Abstracts. Are there any contradictory claims, conflicting findings, or disagreements between the papers?
    2. Second, evaluate the Generated Summary based on how it handles these contradictions.

    Scoring rules:
    - Score 1.0: The summary explicitly notes the disagreement/contradiction, presenting both sides fairly.
    - Score 0.5: The summary vaguely mentions mixed results but fails to clearly articulate the specific contradiction.
    - Score 0.0: The summary cherry-picks one side of the argument and completely ignores the opposing finding present in the sources.
    - Score 1.0: If there are absolutely NO contradictory claims in the Source Abstracts (thus no cherry-picking occurred).

    Return ONLY a valid JSON object. Do not include any other text or markdown formatting outside the JSON.
    Strict Format:
    {{"score": <float_between_0.0_and_1.0>, "reasoning": "<short_explanation_of_found_contradictions_and_handling>"}}
    """

    try:
        response = llm_judge.invoke([HumanMessage(content=eval_prompt)])

        raw_text = response.content.strip().strip('`').replace('json\n', '', 1)
        evaluation = json.loads(raw_text)

        score = float(evaluation.get("score", 1.0))

        print(f"   [Metric] Comparative Coherence: {score}/1.0")
        print(f"   [Metric Reason]: {evaluation.get('reasoning')}")

        return score, evaluation.get('reasoning')

    except Exception as e:
        print(f"⚠⚠️ Error in metric Comparative Coherence: {e}")
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

    eval_prompt = f"""
    You are an expert academic fact-checker checking for "hallucinations" in a research summary.

    Source Text (Ground Truth):
    {source_abstracts}

    Generated Summary:
    {generated_summary}

    Task:
    1. Read the Generated Summary and identify its core factual claims.
    2. Verify if EACH claim is explicitly supported by or directly inferable from the Source Text.
    3. If a claim in the summary is NOT found in the source text, it is a hallucination.
    4. Calculate a faithfulness score from 0.0 to 1.0. (1.0 = entirely grounded/0 hallucinations, 0.5 = half of the claims are unverified, 0.0 = completely fabricated).

    Return ONLY a valid JSON object. Do not include any other text or markdown formatting outside the JSON.
    Strict Format:
    {{"score": <float_between_0.0_and_1.0>, "hallucinations": ["<list_of_unsupported_statements_if_any>"]}}
    """

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

    eval_prompt = f"""
    You are an expert reviewer for a top-tier scientific journal. Your task is to evaluate a research summary for "Methodological Completeness".

    Source Abstracts:
    {source_abstracts}

    Generated Summary:
    {generated_summary}

    Task:
    1. Read the Source Abstracts. Identify if they mention specific Datasets/Sample sizes, Methodologies/Architectures, and Evaluation Metrics.
    2. Check the Generated Summary. Did the summary successfully include these specific methodological details, or did it only state vague general conclusions?
    3. Calculate a score based on how completely the methodological details from the sources are represented in the summary.

    Scoring:
    1.0 = All key methodological details present in the sources are accurately reflected in the summary.
    0.66 = Most details are present, but one key aspect (e.g., the specific dataset name or metric) is missing.
    0.33 = The summary is mostly general conclusions; methodology is severely lacking.
    0.0 = No methodology is mentioned at all despite being in the sources.
    (If the sources themselves contain NO methodological details, score 1.0, as the summary couldn't possibly include them).

    Return ONLY a valid JSON object. Do not include markdown formatting outside the JSON.
    Strict Format:
    {{"score": <float_value>, "missing_elements": ["<list of missed methodological details>"]}}
    """

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

    eval_prompt = f"""
    You are a meticulous data auditor for a scientific institution. Your exact task is to verify the "Statistical & Numerical Factuality" of a research summary.

    Source Abstracts (Ground Truth):
    {source_abstracts}

    Generated Summary:
    {generated_summary}

    Step-by-Step Task:
    1. EXTRACT: Scan the Generated Summary and extract EVERY specific numerical claim, statistic, percentage, or data point mentioned. Count them (total_claims).
    2. VERIFY: Cross-reference each extracted number with the Source Abstracts. Is the context exactly the same?
    3. IDENTIFY ERRORS: Count how many of these numbers are INCORRECT or HALLUCINATED (errors_found). 
    4. CALCULATE: Calculate the exact score using this formula:
       Score = (total_claims - errors_found) / total_claims
       Round to 2 decimal places. (If total_claims is 0, score is 1.0).

    CRITICAL INSTRUCTION FOR JSON OUTPUT:
    Return ONLY a valid JSON object. 
    Inside the "hallucinated_numbers" array, NEVER use double quotes ("). If you need to quote something, use single quotes (') only. Unescaped double quotes will break the parser.

    Strict Format:
    {{
        "total_claims": <int>,
        "errors_found": <int>,
        "score": <float>, 
        "hallucinated_numbers": ["<explanation using single quotes only>"]
    }}
    """

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

    eval_prompt = f"""
    You are a senior meta-analysis researcher evaluating an AI's ability to synthesize conflicting scientific literature.

    Source Abstracts:
    {source_abstracts}

    Generated Summary:
    {generated_summary}

    Task:
    1. Deeply analyze the Source Abstracts for any scientific contradictions, empirical discrepancies, or diverging conclusions between different papers (e.g., Paper A finds a positive correlation, Paper B finds no correlation).
    2. If NO contradictions exist in the sources, automatically award a score of 1.0 and state "No contradictions present."
    3. If contradictions DO exist, evaluate how the Generated Summary handles them:
       - Did it synthesize the debate fairly (e.g., "While Smith found X, Jones demonstrated Y")?
       - Or did it engage in "cherry-picking," presenting only one side as absolute truth and ignoring the conflicting paper?

    Scoring rules (if contradictions exist):
    - 1.0: The summary explicitly and accurately describes the conflicting findings or differing methodologies.
    - 0.5: The summary vaguely hints at "mixed results" but fails to explain what the actual scientific disagreement is.
    - 0.0: The summary completely ignores the contradiction and cherry-picks only one side of the evidence.

    Return ONLY a valid JSON object. Do not include markdown formatting outside the JSON.
    Strict Format:
    {{"score": <float_value>, "explanation": "<Identify the contradiction in sources and explain how the summary handled it>"}}
    """

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

    eval_prompt = f"""
    You are an expert scientific peer-reviewer. Your task is to evaluate a Generated Summary against Source Abstracts based on 6 strict scientific metrics.

    Topic: {topic}

    Source Abstracts:
    {source_abstracts}

    Generated Summary:
    {generated_summary}

    EVALUATION CRITERIA:
    1. Faithfulness (Grounding & Hallucination Check):
    Definition: Every single claim, statement, and conclusion in the Generated Summary MUST be directly traceable to the provided Source Abstracts. The summary must not introduce any external knowledge, assumptions, or logical leaps. If the summary says "A causes B," the source must explicitly state exactly that.
    Scoring: 1.0 if perfectly grounded. Deduct points severely (e.g., -0.2) for every unverified, exaggerated, or hallucinated claim.
    
    2. Key Claim Recall (Core Findings Comprehensiveness):
    Definition: Identify the primary breakthroughs, core arguments, and main conclusions from the Source Abstracts. Evaluate whether the Generated Summary successfully captured these vital points, or if it missed the main message while focusing on trivial background details.
    Scoring: 1.0 if all major findings across the sources are integrated. Deduct points if critical discoveries or main conclusions from the provided abstracts are ignored.
    
    3. Topic Relevance (Focus & Drift Prevention):
    Definition: The summary must rigidly align with the provided Topic. It should not wander into generalized background information, broad historical contexts, or unrelated tangents, even if that extra information was present in the Source Abstracts. Every sentence must serve to answer or explore the specific Topic.
    Scoring: 1.0 if highly focused and concise. Deduct points for fluff, generic introductions, or off-topic paragraphs.
    
    4. Methodological Completeness (Technical Depth):
    Definition: A high-quality scientific summary must explain how the research was conducted, not just the results. Check for the explicit mention of technical details such as: datasets used, model architectures, specific algorithms, baseline comparisons, and evaluation metrics (e.g., precision, F1-score, latency).
    Scoring: 1.0 if the methodology is adequately described. Deduct points (e.g., -0.3) if the summary only lists conclusions without explaining the underlying methods or experimental setups.
    
    5. Statistical & Numerical Factuality (Strict Data Audit):
    Definition: Conduct a rigid numerical audit. Extract EVERY number, percentage, and metric from the Generated Summary and cross-reference it with the Source Abstracts. The context must match perfectly. Claiming "95% accuracy" when the source said "95% recall" is a critical error.
    Scoring: Use this exact formula: (total_numbers_in_summary - mismatched_or_hallucinated_numbers) / total_numbers_in_summary. Note: If the summary contains exactly 0 numerical claims, automatically award a score of 1.0 (no errors).
    
    6. Contradiction Recognition (Scientific Objectivity & Nuance):
    Definition: Scientific papers often disagree or have varying results. If Source A claims a method is highly effective and Source B claims it fails under certain conditions, the summary MUST reflect this debate. The summary should not artificially smooth over conflicting results to create a false consensus. It must also mention key limitations stated by the authors.
    Scoring: 1.0 if nuances, limitations, and contradictions are accurately reported (or if all sources perfectly agree). Deduct points for "cherry-picking" positive results while ignoring negative findings or opposing views present in the sources.
    
    CRITICAL INSTRUCTION:
    Return ONLY a valid JSON object. 
    Use single quotes (') inside string values, NEVER double quotes (") to prevent JSON parsing errors.
    If a score is perfect (1.0), leave the "errors" string empty. If a score is < 1.0, explain exactly what must be fixed in the "errors" string.

    JSON FORMAT:
    {{
        "faithfulness": {{"score": <float>, "errors": "<explanation>"}},
        "key_claim_recall": {{"score": <float>, "errors": "<explanation>"}},
        "topic_relevance": {{"score": <float>, "errors": "<explanation>"}},
        "methodological_completeness": {{"score": <float>, "errors": "<explanation>"}},
        "statistical_factuality": {{"score": <float>, "errors": "<explanation>"}},
        "contradiction_recognition": {{"score": <float>, "errors": "<explanation>"}}
    }}
    """

    try:
        response = llm_judge.invoke([HumanMessage(content=eval_prompt)])
        raw_text = response.content.strip()

        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        clean_json_string = json_match.group(0) if json_match else raw_text

        evaluation = json.loads(clean_json_string)
        return evaluation

    except Exception as e:
        print(f"⚠️ Error of judge: {e}")
        return {metric: {"score": 0.5, "errors": "Evaluation failed, please review draft."}
                for metric in ["faithfulness", "key_claim_recall", "topic_relevance", "methodological_completeness",
                               "statistical_factuality", "contradiction_recognition"]}
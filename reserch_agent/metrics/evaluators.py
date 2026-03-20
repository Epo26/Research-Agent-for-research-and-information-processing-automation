import re
import json
import logging
from langchain_core.messages import HumanMessage
from ..config import PROMPTS

logger = logging.getLogger(__name__)


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
        logger.warning(f"⚠️ Error of judge: {e}")
        return {metric: {"score": 0.0, "errors": "Evaluation failed, please review draft."}
                for metric in ["faithfulness", "key_claim_recall", "topic_relevance", "methodological_completeness",
                               "statistical_factuality", "contradiction_recognition"]}
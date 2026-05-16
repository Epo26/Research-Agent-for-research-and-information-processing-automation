import logging

logger = logging.getLogger(__name__)

def update_token_usage(state: dict, response, model_type: str = "") -> dict:
    """
    Extracts token usage from LLM response and updates the state.
    Optionally tracks tokens for a specific model_type ('cheap', 'smart', 'judge').
    """
    # LangChain Google GenAI usage metadata
    usage = getattr(response, "usage_metadata", {})
    
    if not usage:
        # Fallback for other providers or old versions
        usage = response.response_metadata.get("token_usage", {})

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("candidates_tokens", usage.get("completion_tokens", 0))
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    # Accumulate global tokens
    updates = {
        "total_tokens": state.get("total_tokens", 0) + total_tokens,
        "prompt_tokens": state.get("prompt_tokens", 0) + prompt_tokens,
        "completion_tokens": state.get("completion_tokens", 0) + completion_tokens
    }

    # Accumulate model-specific tokens
    if model_type:
        key = f"tokens_{model_type}"
        updates[key] = state.get(key, 0) + total_tokens

    return updates

import time

def safe_invoke(llm, prompt):
    """
    Invokes the LLM with infinite exponential backoff fault tolerance 
    to handle 503 Service Unavailable and other transient errors.
    """
    delay = 5.0
    attempts = 0
    while True:
        try:
            return llm.invoke(prompt)
        except Exception as e:
            attempts += 1
            logger.warning(f"LLM API Error (Attempt {attempts}): {e}. Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            # Exponential backoff capped at 60 seconds
            delay = min(delay * 1.5, 60.0)

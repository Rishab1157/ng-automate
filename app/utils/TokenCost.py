"""The estimated price of LLM calls, from LiteLLM's price list. Input and output are priced apart."""

import logging

import litellm

from app.models.tokenUsageModel import TokenUsageModel

logger = logging.getLogger(__name__)


def priced(usage: TokenUsageModel, model: str) -> TokenUsageModel:
    """`usage` with its input and output cost for `model` (LiteLLM name). Price one call at a time: some models cost
    more per token above a prompt size, which a sum of calls would wrongly reach. Costs stay None when LiteLLM knows
    no price for the model. Never raises."""
    try:
        input_cost, output_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cached_input_tokens,
            cache_creation_input_tokens=usage.cache_write_tokens,
        )
        return usage.model_copy(update={"input_cost_usd": float(input_cost), "output_cost_usd": float(output_cost)})
    except Exception as error:
        logger.debug("No price for model %s: %s", model, type(error).__name__)
        return usage

"""Tokens the LLM calls used, and their estimated cost: per agent stage, per heal, per test-data reading.

Input and output are priced differently, so they are counted and priced apart.
"""

from pydantic import BaseModel, computed_field

# Below this, a cost is shown with more digits ("~$0.0042" rather than "~$0.00").
MIN_SHORT_COST_USD = 0.01


class TokenUsageModel(BaseModel):
    """Counts as the provider reports them (through LiteLLM)."""

    input_tokens: int = 0  # all prompt tokens, the cached ones included
    cached_input_tokens: int = 0  # input served from the provider's cache (billed cheaper)
    cache_write_tokens: int = 0  # input written to the provider's cache (Anthropic; billed higher)
    output_tokens: int = 0
    reasoning_tokens: int = 0  # part of the output (reasoning models)
    calls: int | None = None  # LLM calls, when known
    # Estimated by LiteLLM from its price list (cached input at its own price). None when no price is known.
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_cost_usd(self) -> float | None:
        return _add_known(self.input_cost_usd, self.output_cost_usd)

    def add(self, other: "TokenUsageModel | None") -> "TokenUsageModel":
        """Both usages together (a count or cost only one side knows is kept)."""
        if other is None:
            return self
        return TokenUsageModel(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            calls=_add_known(self.calls, other.calls),
            input_cost_usd=_add_known(self.input_cost_usd, other.input_cost_usd),
            output_cost_usd=_add_known(self.output_cost_usd, other.output_cost_usd),
        )

    def describe(self) -> str:
        """One line, e.g. "input 1.2M (cached 300k) ~$2.40, output 45k ~$0.45, total ~$2.85" (no costs when the price
        is not known)."""
        text = f"input {_short(self.input_tokens)}"
        if self.cached_input_tokens:
            text += f" (cached {_short(self.cached_input_tokens)})"
        text += _cost(self.input_cost_usd) + f", output {_short(self.output_tokens)}" + _cost(self.output_cost_usd)
        if self.total_cost_usd is not None:
            text += f", total{_cost(self.total_cost_usd)}"
        return text


def _add_known[N: (int, float)](first: N | None, second: N | None) -> N | None:
    if first is None:
        return second
    return first if second is None else first + second


def _short(count: int) -> str:
    """e.g. 950, 45k, 1.2M."""
    for size, unit in ((1_000_000, "M"), (1_000, "k")):
        if count >= size:
            return f"{count / size:.1f}".rstrip("0").rstrip(".") + unit
    return str(count)


def _cost(usd: float | None) -> str:
    if usd is None:
        return ""
    if usd == 0:
        return " $0"
    return f" ~${usd:.2f}" if usd >= MIN_SHORT_COST_USD else f" ~${usd:.4f}"

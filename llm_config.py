"""LLM providers: what differs per provider lives here; the agent itself is shared.

Pick one with the LLM_PROVIDER environment variable (default: groq):

    $env:LLM_PROVIDER = "groq"; python main.py

To add a provider, add an entry to PROVIDERS. Provider-specific model settings
go in its model_kwargs.
"""

import os
from dataclasses import dataclass, field

import groq
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq

DEFAULT_PROVIDER = "groq"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    chat_class: type[BaseChatModel]
    model: str
    # Used for a call when model hits a rate limit. Leave as None to wait and
    # retry instead, which keeps every answer in a run from the same model
    fallback_model: str | None = None
    # Spaces out LLM calls on the client side; None means no throttling
    requests_per_min: float | None = None
    # What this provider's SDK raises on a rate limit (HTTP 429)
    rate_limit_errors: tuple[type[Exception], ...] = ()
    # Passed to chat_class as is: temperature and any provider-specific settings
    model_kwargs: dict = field(default_factory=dict)

    def describe(self) -> dict:
        """The settings as JSON-safe values, saved with each eval run."""
        return {
            "provider": self.name,
            "model": self.model,
            "fallback_model": self.fallback_model,
            "requests_per_min": self.requests_per_min,
            "model_kwargs": self.model_kwargs,
        }


PROVIDERS = {
    "groq": ProviderConfig(
        name="groq",
        chat_class=ChatGroq,
        model="openai/gpt-oss-120b",
        # Groq's rate limits are per model, so the smaller model is often still available
        fallback_model="openai/gpt-oss-20b",
        rate_limit_errors=(groq.RateLimitError,),
        model_kwargs={"temperature": 0},
    ),
}


def get_provider_config() -> ProviderConfig:
    """The provider selected by LLM_PROVIDER. Call after load_dotenv(), so .env can set it."""
    name = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise SystemExit(f"Unknown LLM_PROVIDER '{name}'. Choose from: {', '.join(PROVIDERS)}")
    return PROVIDERS[name]


def _make_chat_model(config: ProviderConfig, model: str, tools: list):
    """bind_tools converts the tool objects to JSON schemas and attaches them to every call."""
    rate_limiter = None
    if config.requests_per_min:
        # max_bucket_size=1 prevents bursts: calls are evenly spaced, even after an idle period
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=config.requests_per_min / 60,
            max_bucket_size=1,
        )
    return config.chat_class(model=model, rate_limiter=rate_limiter, **config.model_kwargs).bind_tools(tools)


def build_llm(config: ProviderConfig, tools: list):
    """The chat model for the agent, with tools bound and the fallback attached if there is one."""
    llm = _make_chat_model(config, config.model, tools)
    if config.fallback_model is None:
        return llm
    # Each call starts on model and switches only on a rate limit. Other errors
    # (bad request, auth) are raised, since another model wouldn't fix them
    return llm.with_fallbacks(
        [_make_chat_model(config, config.fallback_model, tools)],
        exceptions_to_handle=config.rate_limit_errors,
    )

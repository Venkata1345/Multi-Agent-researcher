"""Shared agent abstraction + LLM client.

Every specialist agent is a ``BaseAgent[InputT, OutputT]``: a single async ``run``
mapping one typed Pydantic input to one typed Pydantic output. The shared
structured-output call path lives here, so all agents acquire real generation --
and LangSmith tracing -- through one place.

Provider selection is the *only* place that knows about OpenAI vs Anthropic
(``build_chat_model``); swapping providers is a one-line config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.resources import files
from time import perf_counter
from typing import Generic, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_assistant.config import Settings, get_settings
from research_assistant.errors import AgentError
from research_assistant.messages import A2AMessage
from research_assistant.observability import record_llm_call

InputT = TypeVar("InputT", bound=A2AMessage)
OutputT = TypeVar("OutputT", bound=A2AMessage)
SchemaT = TypeVar("SchemaT", bound=A2AMessage)


def build_chat_model(
    settings: Settings | None = None, *, model: str | None = None
) -> BaseChatModel:
    """Construct a chat model. The provider branch lives only here.

    ``model`` overrides ``settings.research_model`` (used by the eval judge to run
    on a different model than the one being graded).
    """
    settings = settings or get_settings()
    model_name = model or settings.research_model
    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=settings.temperature,
            api_key=settings.openai_api_key,
        )
    if settings.provider == "anthropic":  # pragma: no cover - not wired in v1
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=settings.temperature)
    raise ValueError(f"Unknown provider: {settings.provider!r}")


def load_prompt(name: str) -> str:
    """Load a system-prompt markdown file packaged under ``research_assistant/prompts``."""
    return (
        files("research_assistant.prompts")
        .joinpath(f"{name}.md")
        .read_text(encoding="utf-8")
    )


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Abstract specialist agent.

    Subclasses set ``name`` (LangSmith run tag), ``prompt_name`` (system-prompt
    file stem), and implement ``run``. The model is injectable so tests can pass
    a fake without any network.
    """

    #: Stable identifier used for tracing/logging.
    name: str = "agent"
    #: System-prompt file stem under ``research_assistant/prompts/``.
    prompt_name: str = ""

    def __init__(
        self,
        model: BaseChatModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        # Defer model construction (and its API-key check) until actually needed
        # when no model is injected.
        self._model = model if model is not None else build_chat_model(self._settings)

    @abstractmethod
    async def run(self, payload: InputT) -> OutputT:
        """Execute the agent's single responsibility and return a typed result."""
        raise NotImplementedError

    async def _complete(
        self,
        *,
        user: str,
        schema: type[SchemaT],
        system: str | None = None,
    ) -> SchemaT:
        """One structured-output LLM call, validated into ``schema``.

        Tagged with the agent name so each call shows up as a named node in the
        LangSmith trace tree.
        """
        system_prompt = system if system is not None else load_prompt(self.prompt_name)
        # include_raw=True returns {"raw", "parsed", "parsing_error"} so we can
        # read token usage off the raw message for cost/latency metrics.
        structured = self._model.with_structured_output(
            schema,
            method=self._settings.structured_output_method,
            include_raw=True,
        )
        start = perf_counter()
        result = await structured.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user),
            ],
            config={
                "run_name": self.name,
                "tags": [self.name, "agent"],
                "metadata": {"agent": self.name},
            },
        )
        latency_s = perf_counter() - start

        parsed = result["parsed"]
        if parsed is None:
            raise AgentError(
                f"{self.name}: structured output failed to parse "
                f"({result.get('parsing_error')})"
            )
        usage = getattr(result["raw"], "usage_metadata", None)
        record_llm_call(
            agent=self.name,
            model=self._settings.research_model,
            latency_s=latency_s,
            usage=usage,
        )
        return cast(SchemaT, parsed)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"

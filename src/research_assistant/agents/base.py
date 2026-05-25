"""Shared agent abstraction.

Every specialist agent is a ``BaseAgent[InputT, OutputT]``: a single async
``run`` that maps one typed Pydantic input to one typed Pydantic output. Keeping
the interface this narrow is what makes the agents trivially composable as
LangGraph nodes and trivially mockable in tests.

In Phase 1 the LLM client is absent -- subclasses return hardcoded Pydantic
objects. Phase 2 introduces the shared structured-output LLM client here so all
agents acquire real generation through one code path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from research_assistant.messages import A2AMessage

InputT = TypeVar("InputT", bound=A2AMessage)
OutputT = TypeVar("OutputT", bound=A2AMessage)


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Abstract specialist agent.

    Subclasses set a human-readable ``name`` (used for LangSmith run tags and
    logging in later phases) and implement ``run``.
    """

    #: Stable identifier used for tracing/logging. Overridden per subclass.
    name: str = "agent"

    @abstractmethod
    async def run(self, payload: InputT) -> OutputT:
        """Execute the agent's single responsibility and return a typed result."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"

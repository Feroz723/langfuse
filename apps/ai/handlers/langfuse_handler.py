"""Langfuse evaluation layer integration for QuickVoice AI calls.

This module provides the ``LangfuseCallTracer`` class that instruments every
voice-agent call with Langfuse traces, spans, generations, and evaluation
scores.  The integration is **opt-in** — set ``LANGFUSE_ENABLED=true`` in the
environment to activate it.

Typical lifecycle::

    tracer = LangfuseCallTracer.create(config, call_context)
    tracer.start_session()

    tracer.trace_stt(model="deepgram/nova-3", ...)
    tracer.trace_llm_generation(model="google/gemini-2.5-flash", ...)
    tracer.trace_tts(model="deepgram/aura-2", ...)

    tracer.finalize(transcripts, extracted_data, evaluated_data)
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from utils.logger import logger, redact_sensitive


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def is_langfuse_enabled() -> bool:
    """Return ``True`` when the operator has opted into Langfuse tracing."""
    return os.getenv("LANGFUSE_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_langfuse_client = None


def get_langfuse_client():
    """Return a lazily-initialised Langfuse client singleton.

    Returns ``None`` when Langfuse is disabled or when the SDK is not
    installed.
    """
    global _langfuse_client  # noqa: PLW0603
    if _langfuse_client is not None:
        return _langfuse_client
    if not is_langfuse_enabled():
        return None
    try:
        from langfuse import Langfuse  # type: ignore[import-untyped]

        _langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )
        logger.info("[langfuse] client initialised (host={})", os.getenv("LANGFUSE_BASE_URL", "cloud"))
        return _langfuse_client
    except Exception as exc:
        logger.warning("[langfuse] failed to initialise client: {}", redact_sensitive(str(exc)))
        return None


# ---------------------------------------------------------------------------
# No-op tracer for when Langfuse is disabled
# ---------------------------------------------------------------------------

class _NoOpSpan:
    """Lightweight stub returned when Langfuse is disabled."""

    def end(self, **_kwargs: Any) -> None:  # noqa: D401
        pass

    def update(self, **_kwargs: Any) -> None:
        pass

    def generation(self, **_kwargs: Any) -> "_NoOpSpan":
        return self

    def span(self, **_kwargs: Any) -> "_NoOpSpan":
        return self

    def score(self, **_kwargs: Any) -> None:
        pass


_NOOP_SPAN = _NoOpSpan()


# ---------------------------------------------------------------------------
# LangfuseCallTracer
# ---------------------------------------------------------------------------

class LangfuseCallTracer:
    """Instruments a single QuickVoice call session with Langfuse.

    When Langfuse is disabled (the default), every method is a zero-cost
    no-op so callers never need conditional guards.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        call_context: dict[str, Any],
        client: Any | None = None,
    ) -> None:
        self._config = config
        self._call_context = call_context
        self._client = client
        self._trace: Any = None
        self._session_span: Any = _NOOP_SPAN
        self._enabled = client is not None
        self._start_time: float = time.monotonic()

    # -- Factory ----------------------------------------------------------

    @classmethod
    def create(
        cls,
        config: dict[str, Any],
        call_context: dict[str, Any],
    ) -> "LangfuseCallTracer":
        """Create a tracer, automatically detecting whether Langfuse is on."""
        client = get_langfuse_client()
        return cls(config=config, call_context=call_context, client=client)

    # -- Lifecycle --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_session(self) -> None:
        """Begin a Langfuse trace for this call."""
        if not self._enabled:
            return
        try:
            call_id = self._call_context.get("call_id", "unknown")
            agent_id = self._config.get("agent_id") or self._call_context.get("agent_id", "")
            org_id = self._config.get("organization_id", "")

            self._trace = self._client.trace(
                name="quickvoice-call",
                id=f"qv-{call_id}",
                session_id=f"agent-{agent_id}" if agent_id else None,
                user_id=org_id or None,
                metadata={
                    "call_id": call_id,
                    "agent_id": agent_id,
                    "organization_id": org_id,
                    "direction": self._call_context.get("direction", "inbound"),
                    "from_number": self._call_context.get("from_number", ""),
                    "to_number": self._call_context.get("to_number", ""),
                    "provider": self._call_context.get("provider") or self._config.get("provider", ""),
                    "llm_model": self._config.get("llm_model", ""),
                    "stt_model": self._config.get("stt_model", ""),
                    "tts_model": self._config.get("tts_model", ""),
                },
                tags=["quickvoice", self._call_context.get("direction", "inbound")],
            )
            self._session_span = self._trace.span(
                name="voice-session",
                metadata={"agent_language": self._config.get("agent_language", "en-US")},
            )
            self._start_time = time.monotonic()
            logger.info("[langfuse] trace started for call={}", redact_sensitive(call_id))
        except Exception as exc:
            logger.warning("[langfuse] failed to start trace: {}", redact_sensitive(str(exc)))
            self._enabled = False

    # -- Pipeline step tracers -------------------------------------------

    def trace_stt(
        self,
        *,
        model: str = "",
        language: str = "",
        transcript: str = "",
        duration_ms: float | None = None,
    ) -> None:
        """Record a speech-to-text step."""
        if not self._enabled:
            return
        try:
            span = self._session_span.span(
                name="stt",
                input={"language": language},
                output={"transcript": transcript},
                metadata={"model": model, "duration_ms": duration_ms},
            )
            span.end()
        except Exception as exc:
            logger.debug("[langfuse] stt span failed: {}", str(exc))

    def trace_llm_generation(
        self,
        *,
        model: str = "",
        system_prompt: str = "",
        user_input: str = "",
        agent_output: str = "",
        usage: dict[str, int] | None = None,
    ) -> None:
        """Record an LLM generation (chat turn)."""
        if not self._enabled:
            return
        try:
            gen = self._session_span.generation(
                name="llm-turn",
                model=model,
                input=[
                    {"role": "system", "content": system_prompt[:500]},
                    {"role": "user", "content": user_input},
                ],
                output=agent_output,
                usage=usage or {},
                metadata={"llm_provider": self._config.get("llm_provider", "")},
            )
            gen.end()
        except Exception as exc:
            logger.debug("[langfuse] llm generation span failed: {}", str(exc))

    def trace_tts(
        self,
        *,
        model: str = "",
        voice: str = "",
        text: str = "",
        duration_ms: float | None = None,
    ) -> None:
        """Record a text-to-speech step."""
        if not self._enabled:
            return
        try:
            span = self._session_span.span(
                name="tts",
                input={"text": text},
                metadata={"model": model, "voice": voice, "duration_ms": duration_ms},
            )
            span.end()
        except Exception as exc:
            logger.debug("[langfuse] tts span failed: {}", str(exc))

    def trace_rag_retrieval(
        self,
        *,
        agent_id: str = "",
        query: str = "",
        context: str = "",
        top_k: int = 5,
    ) -> None:
        """Record a RAG / knowledge base retrieval step."""
        if not self._enabled:
            return
        try:
            span = self._session_span.span(
                name="rag-retrieval",
                input={"query": query, "top_k": top_k},
                output={"context": context[:1000] if context else ""},
                metadata={"agent_id": agent_id},
            )
            span.end()
        except Exception as exc:
            logger.debug("[langfuse] rag span failed: {}", str(exc))

    def trace_tool_call(
        self,
        *,
        tool_type: str = "http",
        tool_name: str = "",
        arguments: dict[str, Any] | None = None,
        result: str = "",
        error: str | None = None,
    ) -> None:
        """Record an HTTP or MCP tool invocation."""
        if not self._enabled:
            return
        try:
            span = self._session_span.span(
                name=f"tool-call-{tool_type}",
                input={"tool_name": tool_name, "arguments": arguments or {}},
                output={"result": result[:1000]} if not error else {"error": error},
                metadata={"tool_type": tool_type},
            )
            span.end()
        except Exception as exc:
            logger.debug("[langfuse] tool span failed: {}", str(exc))

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Generator[Any, None, None]:
        """Generic context-manager span for ad-hoc instrumentation."""
        if not self._enabled:
            yield _NOOP_SPAN
            return
        try:
            s = self._session_span.span(name=name, metadata=metadata)
            yield s
            s.end()
        except Exception as exc:
            logger.debug("[langfuse] span '{}' failed: {}", name, str(exc))
            yield _NOOP_SPAN

    # -- Scoring & finalisation ------------------------------------------

    def score_call(
        self,
        *,
        name: str,
        value: float,
        comment: str = "",
    ) -> None:
        """Attach a numeric score to the current call trace."""
        if not self._enabled or self._trace is None:
            return
        try:
            self._trace.score(
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as exc:
            logger.debug("[langfuse] score '{}' failed: {}", name, str(exc))

    def finalize(
        self,
        *,
        transcripts: list[dict[str, Any]] | None = None,
        extracted_data: list[dict[str, Any]] | None = None,
        evaluated_data: list[dict[str, Any]] | None = None,
        duration_seconds: int = 0,
        status: str = "COMPLETED",
    ) -> None:
        """End the session span, compute scores, and flush to Langfuse."""
        if not self._enabled:
            return

        try:
            # -- Duration score --
            self.score_call(
                name="call-duration-seconds",
                value=float(duration_seconds),
                comment=f"Total call length: {duration_seconds}s",
            )

            # -- Transcript turn count --
            transcript_list = transcripts or []
            user_turns = sum(1 for t in transcript_list if t.get("role") == "user")
            agent_turns = sum(1 for t in transcript_list if t.get("role") in ("agent", "assistant"))
            total_turns = user_turns + agent_turns
            self.score_call(
                name="transcript-turns",
                value=float(total_turns),
                comment=f"user={user_turns}, agent={agent_turns}",
            )

            # -- Extracted data completeness --
            data_needed = self._config.get("data_needed") or []
            extracted = extracted_data or []
            if data_needed:
                needed_count = len(data_needed)
                filled_count = sum(
                    1 for item in extracted
                    if item.get("value") not in (None, "")
                )
                completeness = filled_count / needed_count if needed_count > 0 else 1.0
                self.score_call(
                    name="data-extraction-completeness",
                    value=round(completeness, 3),
                    comment=f"{filled_count}/{needed_count} fields collected",
                )

            # -- Evaluation pass rate --
            evaluations = evaluated_data or []
            if evaluations:
                pass_count = sum(
                    1 for e in evaluations
                    if str(e.get("value", "")).strip().lower() in ("true", "yes", "pass", "1")
                )
                pass_rate = pass_count / len(evaluations) if evaluations else 0.0
                self.score_call(
                    name="evaluation-pass-rate",
                    value=round(pass_rate, 3),
                    comment=f"{pass_count}/{len(evaluations)} evaluations passed",
                )

            # -- Call status --
            self.score_call(
                name="call-completed",
                value=1.0 if status == "COMPLETED" else 0.0,
                comment=f"status={status}",
            )

            # -- End session span --
            try:
                self._session_span.end()
            except Exception:
                pass

            # -- Update trace with final output --
            if self._trace is not None:
                try:
                    self._trace.update(
                        output={
                            "status": status,
                            "duration_seconds": duration_seconds,
                            "total_turns": total_turns,
                            "extracted_fields": len(extracted),
                        },
                    )
                except Exception:
                    pass

            # -- Flush --
            self._flush()
            call_id = self._call_context.get("call_id", "unknown")
            logger.info(
                "[langfuse] trace finalised for call={} (turns={}, duration={}s)",
                redact_sensitive(call_id),
                total_turns,
                duration_seconds,
            )
        except Exception as exc:
            logger.warning("[langfuse] finalisation error: {}", redact_sensitive(str(exc)))

    def _flush(self) -> None:
        """Flush pending events to Langfuse."""
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:
            logger.debug("[langfuse] flush failed: {}", str(exc))

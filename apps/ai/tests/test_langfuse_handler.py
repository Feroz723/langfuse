"""Unit tests for handlers/langfuse_handler.py

These tests verify:
1. LangfuseCallTracer is a no-op when Langfuse is disabled
2. Trace, span, generation, and score lifecycle works with a mock client
3. Finalization computes correct evaluation scores
4. Graceful error handling when Langfuse SDK raises exceptions
"""

import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the parent package is importable when running from the repo root.
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from handlers.langfuse_handler import (
    LangfuseCallTracer,
    is_langfuse_enabled,
    get_langfuse_client,
    _NoOpSpan,
)


class TestIsLangfuseEnabled(unittest.TestCase):
    """Test the ``is_langfuse_enabled`` config helper."""

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_langfuse_enabled())

    def test_disabled_when_false(self):
        with patch.dict(os.environ, {"LANGFUSE_ENABLED": "false"}):
            self.assertFalse(is_langfuse_enabled())

    def test_enabled_when_true(self):
        with patch.dict(os.environ, {"LANGFUSE_ENABLED": "true"}):
            self.assertTrue(is_langfuse_enabled())

    def test_enabled_case_insensitive(self):
        for val in ("TRUE", "True", "1", "yes", "on"):
            with patch.dict(os.environ, {"LANGFUSE_ENABLED": val}):
                self.assertTrue(is_langfuse_enabled(), f"failed for {val!r}")


class TestNoOpTracer(unittest.TestCase):
    """When Langfuse is disabled, the tracer should be a zero-cost no-op."""

    def setUp(self):
        self.config = {
            "agent_id": "agent-1",
            "organization_id": "org-1",
            "stt_model": "deepgram/nova-3",
            "llm_model": "google/gemini-2.5-flash",
            "tts_model": "deepgram/aura-2",
            "voice": "asteria",
            "data_needed": [],
        }
        self.call_context = {
            "call_id": "call-123",
            "direction": "inbound",
        }
        self.tracer = LangfuseCallTracer(
            config=self.config,
            call_context=self.call_context,
            client=None,
        )

    def test_disabled_flag(self):
        self.assertFalse(self.tracer.enabled)

    def test_start_session_noop(self):
        # Should not raise
        self.tracer.start_session()

    def test_trace_methods_noop(self):
        self.tracer.trace_stt(model="test", transcript="hello")
        self.tracer.trace_llm_generation(model="test", user_input="hi", agent_output="hello")
        self.tracer.trace_tts(model="test", text="hello")
        self.tracer.trace_rag_retrieval(query="test")
        self.tracer.trace_tool_call(tool_name="test")

    def test_score_noop(self):
        self.tracer.score_call(name="test", value=1.0)

    def test_finalize_noop(self):
        self.tracer.finalize(duration_seconds=10)


class TestMockTracer(unittest.TestCase):
    """Test tracer with a mock Langfuse client."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_trace = MagicMock()
        self.mock_session_span = MagicMock()
        self.mock_generation = MagicMock()

        self.mock_client.trace.return_value = self.mock_trace
        self.mock_trace.span.return_value = self.mock_session_span
        self.mock_session_span.span.return_value = MagicMock()
        self.mock_session_span.generation.return_value = self.mock_generation

        self.config = {
            "agent_id": "agent-42",
            "organization_id": "org-7",
            "stt_model": "deepgram/nova-3",
            "llm_model": "google/gemini-2.5-flash",
            "llm_provider": "google",
            "tts_model": "deepgram/aura-2",
            "voice": "asteria",
            "agent_language": "en-US",
            "data_needed": [
                {"name": "customer_name"},
                {"name": "email"},
                {"name": "phone"},
            ],
        }
        self.call_context = {
            "call_id": "call-abc-123",
            "direction": "inbound",
            "from_number": "+1234567890",
            "to_number": "+0987654321",
        }
        self.tracer = LangfuseCallTracer(
            config=self.config,
            call_context=self.call_context,
            client=self.mock_client,
        )

    def test_enabled(self):
        self.assertTrue(self.tracer.enabled)

    def test_start_session_creates_trace(self):
        self.tracer.start_session()
        self.mock_client.trace.assert_called_once()
        call_kwargs = self.mock_client.trace.call_args[1]
        self.assertEqual(call_kwargs["name"], "quickvoice-call")
        self.assertEqual(call_kwargs["id"], "qv-call-abc-123")
        self.assertIn("inbound", call_kwargs["tags"])

    def test_start_session_creates_session_span(self):
        self.tracer.start_session()
        self.mock_trace.span.assert_called_once()

    def test_trace_stt(self):
        self.tracer.start_session()
        self.tracer.trace_stt(
            model="deepgram/nova-3",
            language="en-US",
            transcript="Hello, how can I help?",
        )
        self.mock_session_span.span.assert_called()
        call_kwargs = self.mock_session_span.span.call_args[1]
        self.assertEqual(call_kwargs["name"], "stt")

    def test_trace_llm_generation(self):
        self.tracer.start_session()
        self.tracer.trace_llm_generation(
            model="google/gemini-2.5-flash",
            system_prompt="You are helpful.",
            user_input="What time is it?",
            agent_output="I don't have access to the current time.",
        )
        self.mock_session_span.generation.assert_called()
        call_kwargs = self.mock_session_span.generation.call_args[1]
        self.assertEqual(call_kwargs["name"], "llm-turn")
        self.assertEqual(call_kwargs["model"], "google/gemini-2.5-flash")

    def test_trace_tts(self):
        self.tracer.start_session()
        self.tracer.trace_tts(model="deepgram/aura-2", voice="asteria", text="Hello!")
        call_kwargs = self.mock_session_span.span.call_args[1]
        self.assertEqual(call_kwargs["name"], "tts")

    def test_trace_rag_retrieval(self):
        self.tracer.start_session()
        self.tracer.trace_rag_retrieval(
            agent_id="agent-42",
            query="pricing info",
            context="Our pricing starts at $10/month.",
        )
        call_kwargs = self.mock_session_span.span.call_args[1]
        self.assertEqual(call_kwargs["name"], "rag-retrieval")

    def test_trace_tool_call(self):
        self.tracer.start_session()
        self.tracer.trace_tool_call(
            tool_type="http",
            tool_name="get_weather",
            arguments={"city": "NYC"},
            result='{"temp": 72}',
        )
        call_kwargs = self.mock_session_span.span.call_args[1]
        self.assertEqual(call_kwargs["name"], "tool-call-http")

    def test_score_call(self):
        self.tracer.start_session()
        self.tracer.score_call(name="quality", value=0.9, comment="good")
        self.mock_trace.score.assert_called_once_with(
            name="quality",
            value=0.9,
            comment="good",
        )

    def test_finalize_computes_scores(self):
        self.tracer.start_session()
        transcripts = [
            {"role": "user", "message": "Hi"},
            {"role": "agent", "message": "Hello!"},
            {"role": "user", "message": "What's your pricing?"},
            {"role": "agent", "message": "We start at $10/month."},
        ]
        extracted = [
            {"name": "customer_name", "value": "John"},
            {"name": "email", "value": "john@example.com"},
            {"name": "phone", "value": ""},
        ]

        self.tracer.finalize(
            transcripts=transcripts,
            extracted_data=extracted,
            duration_seconds=45,
            status="COMPLETED",
        )

        # Check that multiple scores were attached
        score_calls = self.mock_trace.score.call_args_list
        score_names = [call[1]["name"] for call in score_calls]
        self.assertIn("call-duration-seconds", score_names)
        self.assertIn("transcript-turns", score_names)
        self.assertIn("data-extraction-completeness", score_names)
        self.assertIn("call-completed", score_names)

        # Duration score
        duration_call = next(c for c in score_calls if c[1]["name"] == "call-duration-seconds")
        self.assertEqual(duration_call[1]["value"], 45.0)

        # Transcript turns = 4 total (2 user + 2 agent)
        turns_call = next(c for c in score_calls if c[1]["name"] == "transcript-turns")
        self.assertEqual(turns_call[1]["value"], 4.0)

        # Data extraction: 2 out of 3 fields filled = 0.667
        extraction_call = next(c for c in score_calls if c[1]["name"] == "data-extraction-completeness")
        self.assertAlmostEqual(extraction_call[1]["value"], 0.667, places=3)

        # Flush called
        self.mock_client.flush.assert_called()

    def test_finalize_with_evaluations(self):
        self.tracer.start_session()
        evaluations = [
            {"name": "appointment_booked", "value": "true"},
            {"name": "satisfied", "value": "yes"},
            {"name": "transferred", "value": "false"},
        ]
        self.tracer.finalize(
            evaluated_data=evaluations,
            duration_seconds=30,
        )
        score_calls = self.mock_trace.score.call_args_list
        pass_rate_call = next(
            (c for c in score_calls if c[1]["name"] == "evaluation-pass-rate"),
            None,
        )
        self.assertIsNotNone(pass_rate_call)
        # 2 out of 3 = 0.667
        self.assertAlmostEqual(pass_rate_call[1]["value"], 0.667, places=3)


class TestGracefulErrors(unittest.TestCase):
    """Test that Langfuse errors never crash the voice session."""

    def test_start_session_error_disables(self):
        mock_client = MagicMock()
        mock_client.trace.side_effect = RuntimeError("connection failed")
        tracer = LangfuseCallTracer(
            config={"agent_id": "a"},
            call_context={"call_id": "c"},
            client=mock_client,
        )
        tracer.start_session()
        self.assertFalse(tracer.enabled)

    def test_span_error_does_not_propagate(self):
        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_span.span.side_effect = RuntimeError("span error")
        mock_client.trace.return_value = mock_trace
        mock_trace.span.return_value = mock_span

        tracer = LangfuseCallTracer(
            config={"agent_id": "a"},
            call_context={"call_id": "c"},
            client=mock_client,
        )
        tracer.start_session()
        # These should not raise
        tracer.trace_stt(transcript="hello")
        tracer.trace_tts(text="hello")
        tracer.trace_rag_retrieval(query="test")


class TestContextManagerSpan(unittest.TestCase):
    """Test the generic context-manager span."""

    def test_noop_span_when_disabled(self):
        tracer = LangfuseCallTracer(
            config={}, call_context={}, client=None,
        )
        with tracer.span("test") as s:
            self.assertIsInstance(s, _NoOpSpan)

    def test_span_with_mock_client(self):
        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_session_span = MagicMock()
        inner_span = MagicMock()
        mock_client.trace.return_value = mock_trace
        mock_trace.span.return_value = mock_session_span
        mock_session_span.span.return_value = inner_span

        tracer = LangfuseCallTracer(
            config={"agent_id": "a"},
            call_context={"call_id": "c"},
            client=mock_client,
        )
        tracer.start_session()
        with tracer.span("custom-step", extra="data") as s:
            self.assertEqual(s, inner_span)
        inner_span.end.assert_called_once()


if __name__ == "__main__":
    unittest.main()

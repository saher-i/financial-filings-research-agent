"""HTTP chat contract and customer route checks without model calls."""

import io
import json
import os
import unittest
from unittest.mock import patch

from devui.server import Handler
from src.answer_agent.models import AFFORDABLE_MODEL, QUALITY_CANDIDATE_MODEL
from src.answer_agent.provider import DEFAULT_MODEL


class Probe(Handler):
    def __init__(self, path, body=None):
        self.path = path
        self.headers = {"Host": "localhost:8000"}
        self.payload = body
        self.reply = None
        self.wfile = io.BytesIO()

    def _body(self):
        return self.payload

    def _json(self, status, payload):
        self.reply = (status, payload)

    def _file(self, path, kind):
        self.reply = (200, {"file": path.name, "type": kind})

    def _hidden(self):
        self.reply = (404, {"error": "Not found"})


class StreamProbe(Probe):
    def send_response(self, status):
        self.response_status = status

    def send_header(self, name, value):
        pass

    def end_headers(self):
        pass


class CustomerChatTests(unittest.TestCase):
    def test_normal_chat_enables_semantic_verification(self):
        with patch("src.answer_agent.graph.answer_question", return_value={"answer": "Reviewed."}) as answerer:
            from devui.server import run_chat
            run_chat("What changed?")
        self.assertIs(answerer.call_args.kwargs["verify"], True)

    def test_chat_model_selection_reaches_agent(self):
        with patch.dict(os.environ, {"APP_MODE": "customer", "FIREWORKS_API_KEY": "test-key"}), \
             patch("devui.server.run_chat", return_value={"answer": "Supported [S1].",
                                                         "model": AFFORDABLE_MODEL, "sources": [], "trace": []}) as answerer, \
             patch("devui.observability.record_run", return_value="test-run"):
            listing = Probe("/api/chat/models")
            listing.do_GET()
            self.assertEqual(listing.reply[0], 200)
            self.assertEqual(listing.reply[1]["default"], DEFAULT_MODEL)
            self.assertIn(AFFORDABLE_MODEL, {item["id"] for item in listing.reply[1]["models"]})
            self.assertIn(QUALITY_CANDIDATE_MODEL, {item["id"] for item in listing.reply[1]["models"]})
            request = Probe("/api/chat", {"question": "Revenue?", "model": AFFORDABLE_MODEL})
            request.do_POST()
            self.assertEqual(request.reply[0], 200)
            self.assertEqual(answerer.call_args.kwargs["model"], AFFORDABLE_MODEL)
            self.assertEqual(request.reply[1]["model"], AFFORDABLE_MODEL)

    def test_customer_page_and_assets_work_without_developer_mode(self):
        with patch.dict(os.environ, {"APP_MODE": "customer"}):
            for route, name in (("/", "chat.html"), ("/chat.css", "chat.css"),
                                ("/chat.js", "chat.js")):
                probe = Probe(route)
                probe.do_GET()
                self.assertEqual(probe.reply[0], 200)
                self.assertEqual(probe.reply[1]["file"], name)
            hidden = Probe("/ingestion-pipeline")
            hidden.do_GET()
            self.assertEqual(hidden.reply[0], 404)
        with patch.dict(os.environ, {"APP_MODE": "developer"}):
            public = Probe("/")
            public.do_GET()
            self.assertEqual(public.reply[1]["file"], "chat.html")
            developer = Probe("/ingestion-pipeline")
            developer.do_GET()
            self.assertEqual(developer.reply[1]["file"], "ingestion-pipeline.html")
            theme = Probe("/dev/dev-theme.css")
            theme.do_GET()
            self.assertEqual(theme.reply[1]["file"], "dev-theme.css")
            removed = Probe("/pipeline-evaluation")
            removed.do_GET()
            self.assertEqual(removed.reply[0], 404)

    def test_chat_returns_top_level_answer_and_sources(self):
        payload = {"answer": "Apple revenue was $416.2 billion [S1].",
                   "sources": [{"type": "sql", "id": "[S1]", "sql": "SELECT revenue ..."}],
                   "trace": [{"tool": "sql"}],
                   "verification": {"status": "supported"}}
        with patch.dict(os.environ, {"APP_MODE": "customer", "FIREWORKS_API_KEY": "test-key"}), \
             patch("devui.server.run_chat", return_value=payload) as answerer, \
             patch("devui.observability.record_run", return_value="test-run"):
            probe = Probe("/api/chat", {"question": "  What was Apple's revenue?  "})
            probe.do_POST()
            self.assertEqual(probe.reply[0], 200)
            self.assertEqual({key: probe.reply[1][key] for key in payload}, payload)
            self.assertGreaterEqual(probe.reply[1]["total_ms"], 0)
            self.assertEqual(answerer.call_args.args, ("What was Apple's revenue?",))
            self.assertTrue(callable(answerer.call_args.kwargs["progress"]))
            self.assertEqual(answerer.call_args.kwargs["answer_style"], "concise")
            json.dumps(probe.reply[1])

    def test_invalid_question_and_missing_key(self):
        with patch.dict(os.environ, {"APP_MODE": "customer", "FIREWORKS_API_KEY": "test-key"}):
            for invalid in (None, 42, "  ", "x" * 2001):
                probe = Probe("/api/chat", {"question": invalid})
                probe.do_POST()
                self.assertEqual(probe.reply[0], 400)
            bad_style = Probe("/api/chat", {"question": "Revenue?", "answer_style": "unbounded"})
            bad_style.do_POST()
            self.assertEqual(bad_style.reply[0], 400)
        with patch.dict(os.environ, {"APP_MODE": "customer"}, clear=True):
            probe = Probe("/api/chat", {"question": "Valid question"})
            probe.do_POST()
            self.assertEqual(probe.reply[0], 503)

    def test_stream_reports_real_agent_progress_then_answer(self):
        def answerer(question, progress=None, answer_style="concise", model=None):
            self.assertEqual(question, "Revenue?")
            self.assertEqual(answer_style, "detailed")
            progress({"id": "tool-1-c1", "phase": "start", "stage": "sql",
                      "message": "Checking financial figures"})
            progress({"id": "tool-1-c1", "phase": "complete", "stage": "sql",
                      "message": "Checking financial figures", "duration_ms": 24.5})
            return {"answer": "$416.2 billion [S1].", "sources": [], "trace": [],
                    "answer_status": "answered"}

        with patch.dict(os.environ, {"APP_MODE": "customer", "FIREWORKS_API_KEY": "test-key"}), \
             patch("devui.server.run_chat", side_effect=answerer), \
             patch("devui.observability.record_run", return_value="test-run"):
            probe = StreamProbe("/api/chat/stream", {"question": "Revenue?", "answer_style": "detailed"})
            probe.do_POST()
            data = probe.wfile.getvalue().decode()
            self.assertEqual(probe.response_status, 200)
            self.assertIn('event: progress\ndata: {"id": "tool-1-c1", "phase": "start"', data)
            self.assertIn('"duration_ms": 24.5', data)
            self.assertIn('event: answer\ndata: {"answer": "$416.2 billion [S1]."', data)
            self.assertIn('"total_ms":', data)

    def test_pdf_route_only_serves_manifest_filing(self):
        with patch.dict(os.environ, {"APP_MODE": "customer"}):
            known = Probe("/filings/AAPL_FY2025_10-K.pdf")
            known.do_GET()
            self.assertEqual(known.reply[0], 200)
            self.assertEqual(known.reply[1]["type"], "application/pdf")
            unknown = Probe("/filings/%2E%2E%2F.env")
            unknown.do_GET()
            self.assertEqual(unknown.reply[0], 404)

    def test_recent_chats_and_saved_answer_routes(self):
        saved = {"id": "012345abcdef", "question": "Revenue?", "answer": "$100 [S1].",
                 "sources": [{"id": "S1", "type": "sql"}], "total_ms": 23.0}
        with patch("devui.observability.recent_chats", return_value=[
                {"id": saved["id"], "question": saved["question"]}]), \
             patch("devui.observability.chat_by_id", return_value=saved):
            history = Probe("/api/chat/history")
            history.do_GET()
            self.assertEqual(history.reply[1]["chats"][0]["question"], "Revenue?")
            detail = Probe("/api/chat/history/012345abcdef")
            detail.do_GET()
            self.assertEqual(detail.reply[1]["chat"]["answer"], "$100 [S1].")
            invalid = Probe("/api/chat/history/../.env")
            invalid.do_GET()
            self.assertEqual(invalid.reply[0], 404)

    def test_followup_passes_saved_context_and_delete_is_scoped(self):
        history = [{"role": "user", "content": "Apple revenue?"},
                   {"role": "assistant", "content": "$416B."}]
        with patch.dict(os.environ, {"APP_MODE": "customer", "FIREWORKS_API_KEY": "test-key"}), \
             patch("devui.observability.conversation_context", return_value=history), \
             patch("devui.server.run_chat", return_value={"answer": "It rose [S1].", "sources": [], "trace": []}) as answerer, \
             patch("devui.observability.record_run", return_value="123456abcdef") as recorder, \
             patch("devui.observability.delete_conversation", return_value=True) as remover, \
             patch("devui.observability.conversation_by_id", return_value={
                 "id": "aaaabbbbcccc", "turns": [{"question": "Apple revenue?"},
                                                  {"question": "How much did it grow?"}]}):
            request = Probe("/api/chat", {"question": "How much did it grow?",
                                          "conversation_id": "aaaabbbbcccc"})
            request.do_POST()
            self.assertEqual(request.reply[0], 200)
            self.assertEqual(answerer.call_args.kwargs["history"], history)
            self.assertEqual(request.reply[1]["conversation_id"], "aaaabbbbcccc")
            self.assertEqual(recorder.call_args.kwargs["conversation_id"], "aaaabbbbcccc")
            saved = Probe("/api/chat/conversations/aaaabbbbcccc")
            saved.do_GET()
            self.assertEqual(len(saved.reply[1]["conversation"]["turns"]), 2)
            delete = Probe("/api/chat/conversations/aaaabbbbcccc")
            delete.do_DELETE()
            self.assertEqual(delete.reply, (200, {"deleted": True}))
            remover.assert_called_once_with("aaaabbbbcccc")
            malformed = Probe("/api/chat", {"question": "Revenue?", "conversation_id": "../bad"})
            malformed.do_POST()
            self.assertEqual(malformed.reply[0], 400)

    def test_origin_mismatch_is_rejected_before_model_call(self):
        with patch.dict(os.environ, {"APP_MODE": "customer", "FIREWORKS_API_KEY": "test-key"}), \
             patch("devui.server.run_chat") as answerer:
            probe = Probe("/api/chat", {"question": "Revenue?"})
            probe.headers["Origin"] = "https://unrelated.example"
            probe.do_POST()
            self.assertEqual(probe.reply[0], 403)
            answerer.assert_not_called()

    def test_removed_pipeline_trial_route_is_hidden(self):
        with patch.dict(os.environ, {"APP_MODE": "developer", "FIREWORKS_API_KEY": "test-key"}):
            probe = Probe("/api/dev/pipeline-evaluation/run", {
                "strategy": "structure", "target": 3600, "maximum": 4800, "overlap": 3600})
            probe.do_POST()
            self.assertEqual(probe.reply[0], 404)


if __name__ == "__main__":
    unittest.main()

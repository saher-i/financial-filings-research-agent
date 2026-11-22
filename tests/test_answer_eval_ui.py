"""Benchmark answer generation stays gated and starts a bounded run."""

import json
import os
import unittest
from unittest.mock import patch

from devui import server
from tests.test_chat_api import Probe


class BenchmarkGenerationRouteTests(unittest.TestCase):
    def test_page_is_developer_only(self):
        with patch.dict(os.environ, {"APP_MODE": "customer"}):
            probe = Probe("/benchmark-evaluation")
            probe.do_GET()
            self.assertEqual(probe.reply[0], 404)
            generation = Probe("/generation-evaluation")
            generation.do_GET()
            self.assertEqual(generation.reply[0], 404)
        with patch.dict(os.environ, {"APP_MODE": "developer"}):
            probe = Probe("/benchmark-evaluation")
            probe.do_GET()
            self.assertEqual(probe.reply[1]["file"], "benchmark-evaluation.html")
            removed = Probe("/answer-evaluation")
            removed.do_GET()
            self.assertEqual(removed.reply[0], 404)
            generation = Probe("/generation-evaluation")
            generation.do_GET()
            self.assertEqual(generation.reply[1]["file"], "generation-evaluation.html")

    def test_generation_source_preview_is_developer_only(self):
        with patch.dict(os.environ, {"APP_MODE": "customer"}):
            hidden = Probe("/api/dev/generation-evaluation/source?question_id=q_012&source_id=S1")
            hidden.do_GET()
            self.assertEqual(hidden.reply[0], 404)
        with patch.dict(os.environ, {"APP_MODE": "developer"}), \
             patch("eval_pipeline.generation.saved_cited_source", return_value={"id": "S1", "type": "sql"}):
            preview = Probe("/api/dev/generation-evaluation/source?question_id=q_012&source_id=S1")
            preview.do_GET()
            self.assertEqual(preview.reply, (200, {"source": {"id": "S1", "type": "sql"}}))

    def test_run_requires_index_and_accepts_one_job(self):
        with patch.dict(os.environ, {"APP_MODE": "developer", "FIREWORKS_API_KEY": "test-key"}), \
             patch("devui.server.vector_state", return_value={"index": {"ready": True}}), \
             patch("devui.server.threading.Thread") as thread:
            try:
                probe = Probe("/api/dev/benchmark-evaluation/generate", {})
                probe.do_POST()
                self.assertEqual(probe.reply[0], 202)
                self.assertTrue(server.CHAT_LOCK.locked())
                thread.assert_called_once()
                duplicate = Probe("/api/dev/benchmark-evaluation/generate", {})
                duplicate.do_POST()
                self.assertEqual(duplicate.reply[0], 409)
            finally:
                if server.CHAT_LOCK.locked():
                    server.CHAT_LOCK.release()
                server.ANSWER_RUN_JOB = {"state": "idle", "message": ""}

    def test_model_discovery_and_selected_eval_model(self):
        model = "accounts/fireworks/models/tool-model"
        with patch.dict(os.environ, {"APP_MODE": "developer", "FIREWORKS_API_KEY": "test-key"}), \
             patch("src.answer_agent.models.discover_models", return_value=[{"id": model, "label": "Tool model"}]), \
             patch("devui.server.vector_state", return_value={"index": {"ready": True}}), \
             patch("devui.server.threading.Thread") as thread:
            try:
                refresh = Probe("/api/dev/models/refresh", {})
                refresh.do_POST()
                self.assertEqual(refresh.reply[0], 200)
                listing = Probe("/api/dev/models")
                listing.do_GET()
                self.assertIn(model, {item["id"] for item in listing.reply[1]["models"]})
                run = Probe("/api/dev/benchmark-evaluation/generate", {"model": model})
                run.do_POST()
                self.assertEqual(run.reply[0], 202)
                self.assertEqual(thread.call_args.kwargs["args"], (model,))
            finally:
                if server.CHAT_LOCK.locked():
                    server.CHAT_LOCK.release()
                server.ANSWER_RUN_JOB = {"state": "idle", "message": ""}
                server.MODEL_CATALOG.clear()

    def test_benchmark_generation_reviews_every_answer(self):
        def fake_run(answerer, **kwargs):
            answerer("Question")
            self.assertTrue(kwargs["preserve_previous"])
            return {"completed": 1, "total": 1}

        with patch("eval_pipeline.answers.run_development", side_effect=fake_run), \
             patch("devui.server.run_chat", return_value={"answer": "Reviewed"}) as answerer:
            server.CHAT_LOCK.acquire()
            server.run_benchmark_answers("test-model")
            answerer.assert_called_once_with("Question", model="test-model", verify=True)
            self.assertFalse(server.CHAT_LOCK.locked())

    def test_retry_one_answer_starts_verified_agent_without_grading(self):
        rows = [{"id": f"q_{index:03}"} for index in range(1, 11)]
        with patch.dict(os.environ, {"APP_MODE": "developer", "FIREWORKS_API_KEY": "test-key"}), \
             patch("devui.server.evaluation_model", return_value="test-model"), \
             patch("devui.server.threading.Thread") as thread, \
             patch("pathlib.Path.read_text", return_value=json.dumps({"questions": rows})):
            try:
                probe = Probe("/api/dev/benchmark-evaluation/retry-answer",
                              {"question_id": "q_003", "model": "test-model"})
                probe.do_POST()
                self.assertEqual(probe.reply[0], 202)
                self.assertEqual(thread.call_args.kwargs["args"], ("q_003", "test-model"))
            finally:
                if server.CHAT_LOCK.locked():
                    server.CHAT_LOCK.release()
                server.ANSWER_RUN_JOB = {"state": "idle", "message": ""}


if __name__ == "__main__":
    unittest.main()

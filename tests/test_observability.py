"""Local workflow records contain actual steps and remain developer-only."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from devui.observability import (chat_by_id, conversation_by_id, conversation_context,
                                 delete_conversation, recent_chats, recent_runs, record_run)
from test_chat_api import Probe


class WorkflowTests(unittest.TestCase):
    def test_completed_run_can_be_inspected_without_prompts_or_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            events = [{"id": "model-1", "phase": "start", "stage": "planning",
                       "message": "Choosing sources"},
                      {"id": "model-1", "phase": "complete", "stage": "planning",
                       "message": "Choosing sources", "duration_ms": 42.0}]
            result = {"answer": "Revenue was 100 [S1].", "answer_status": "answered",
                      "trace": [{"tool": "query_financials", "arguments": {"sql": "SELECT revenue FROM income_statements"},
                                 "source_ids": ["S1"]}],
                      "sources": [{"id": "S1", "type": "sql", "rows": [{"revenue": 100}]}]}
            run_id = record_run("Revenue?", events, result, 53.1, path)
            saved = recent_runs(path=path)[0]
            self.assertEqual(saved["id"], run_id)
            self.assertEqual(saved["events"], events)
            self.assertEqual(saved["trace"], result["trace"])
            self.assertEqual(recent_chats(path=path)[0]["question"], "Revenue?")
            self.assertEqual(chat_by_id(run_id, path)["answer"], result["answer"])
            self.assertNotIn("messages", saved)
            self.assertNotIn("api_key", path.read_text().lower())

    def test_workflow_page_and_data_are_developer_only(self):
        with patch("devui.observability.recent_runs", return_value=[]):
            with patch.dict("os.environ", {"APP_MODE": "customer"}):
                for path in ("/agent-workflow", "/api/dev/agent-workflow"):
                    probe = Probe(path)
                    probe.do_GET()
                    self.assertEqual(probe.reply[0], 404)
            with patch.dict("os.environ", {"APP_MODE": "developer"}):
                page = Probe("/agent-workflow")
                page.do_GET()
                self.assertEqual(page.reply[1]["file"], "agent-workflow.html")
                data = Probe("/api/dev/agent-workflow")
                data.do_GET()
                self.assertEqual(data.reply, (200, {"runs": []}))

    def test_conversation_restores_followups_and_delete_removes_only_selected_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            first = record_run("Apple revenue?", [], {"answer": "$416B [S1].", "sources": []},
                               20, path, conversation_id="aaaabbbbcccc")
            record_run("What about last year?", [], {"answer": "$391B [S2].", "sources": []},
                       25, path, conversation_id="aaaabbbbcccc")
            other = record_run("Microsoft revenue?", [], {"answer": "$282B [S1].", "sources": []},
                               30, path, conversation_id="ddddeeeeffff")
            self.assertEqual(len(recent_chats(path=path)), 2)
            chat = conversation_by_id("aaaabbbbcccc", path)
            self.assertEqual([turn["question"] for turn in chat["turns"]],
                             ["Apple revenue?", "What about last year?"])
            self.assertEqual(conversation_context("aaaabbbbcccc", path)[-1]["content"], "$391B .")
            self.assertTrue(delete_conversation("aaaabbbbcccc", path))
            self.assertIsNone(chat_by_id(first, path))
            self.assertIsNone(conversation_by_id("aaaabbbbcccc", path))
            self.assertIsNotNone(chat_by_id(other, path))


if __name__ == "__main__":
    unittest.main()

import io
import json
import unittest
from unittest.mock import patch

from src.answer_agent.models import AFFORDABLE_MODEL, QUALITY_CANDIDATE_MODEL, discover_models, validate_model
from devui.server import evaluation_model


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ModelCatalogTests(unittest.TestCase):
    def test_discovery_keeps_only_serverless_tool_models(self):
        requested = []

        def fetch(request, timeout):
            requested.append(request.full_url)
            self.assertEqual(timeout, 15)
            self.assertTrue(request.get_header("Authorization").startswith("Bearer "))
            return Response(json.dumps({"models": [
                {"name": "accounts/fireworks/models/tool-model", "displayName": "Tool model",
                 "supportsServerless": True, "supportsTools": True},
                {"name": "accounts/fireworks/models/embed-model", "supportsServerless": True,
                 "supportsTools": False},
            ]}).encode())

        self.assertEqual([item["id"] for item in discover_models("test-key", fetch)],
                         ["accounts/fireworks/models/tool-model"])
        self.assertIn("pageSize=200", requested[0])

    def test_selection_requires_discovered_model(self):
        self.assertEqual(evaluation_model(AFFORDABLE_MODEL), AFFORDABLE_MODEL)
        self.assertEqual(evaluation_model(QUALITY_CANDIDATE_MODEL), QUALITY_CANDIDATE_MODEL)
        with patch("devui.server.MODEL_CATALOG", [{"id": "accounts/fireworks/models/tool-model"}]):
            self.assertEqual(evaluation_model("accounts/fireworks/models/tool-model"),
                             "accounts/fireworks/models/tool-model")
            with self.assertRaises(ValueError):
                evaluation_model("accounts/fireworks/models/unknown-model")
        with self.assertRaises(ValueError):
            validate_model("http://untrusted.example")


if __name__ == "__main__":
    unittest.main()

"""Query reconstruction must preserve the requested company and period."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.retrieval_pipeline.rewrite import _clean, _known_entities, rewrite_lexical_query


class RewriteTests(unittest.TestCase):
    def test_rejects_rewrite_that_drops_requested_period_or_company(self):
        original = "Apple FY2025 YouTu advertising revenue"
        self.assertEqual(_clean("YouTube ads revenue", original), original)
        self.assertEqual(_clean("Apple FY2024 YouTube ads revenue", original), original)
        self.assertEqual(_clean("Apple FY2025 YouTube ads revenue", original),
                         "Apple FY2025 YouTube ads revenue")

    def test_uses_available_company_metadata_for_unseen_names(self):
        with patch("src.retrieval_pipeline.rewrite.get_schema", return_value={"companies": [
                {"ticker": "NVDA", "name": "Nvidia Corporation"}] }):
            _known_entities.cache_clear()
            self.assertEqual(_clean("FY2025 revenue", "Nvidia FY2025 revenue"),
                             "Nvidia FY2025 revenue")
            self.assertEqual(_clean("NVDA FY2025 revenue", "NVDA FY2025 revenue"),
                             "NVDA FY2025 revenue")
            self.assertEqual(_clean("FY2025 revenue", "NVDA FY2025 revenue"),
                             "NVDA FY2025 revenue")
        _known_entities.cache_clear()

    def test_does_not_treat_leading_article_as_company_alias(self):
        with patch("src.retrieval_pipeline.rewrite.get_schema", return_value={"companies": [
                {"ticker": "ABC", "name": "The Example Company"}]}):
            _known_entities.cache_clear()
            self.assertEqual(_clean("The filing's revenue", "The filing's revenue change",
                                    entities=_known_entities()), "The filing's revenue")
            self.assertEqual(_clean("FY2025 revenue", "The Example Company FY2025 revenue",
                                    entities=_known_entities()), "The Example Company FY2025 revenue")
        _known_entities.cache_clear()

    def test_preserves_unindexed_uppercase_identifier(self):
        self.assertEqual(_clean("FY2025 revenue", "ACME FY2025 revenue", entities=()),
                         "ACME FY2025 revenue")

    def test_short_model_response_and_failure_fallback(self):
        calls = []
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: (calls.append(kwargs), SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="YouTube advertising revenue"))]))[1])))
        result = rewrite_lexical_query("YouTu advertising revenue", client=client)
        self.assertEqual(result["query"], "YouTube advertising revenue")
        self.assertEqual(calls[0]["max_tokens"], 80)
        self.assertEqual(rewrite_lexical_query("YouTu ads", client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: 1 / 0))))["status"],
            "fallback")


if __name__ == "__main__":
    unittest.main()

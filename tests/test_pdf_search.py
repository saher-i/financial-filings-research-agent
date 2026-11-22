"""The agent's filing tool passes through the general hybrid ranking."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.answer_agent.tools.filings import _pdf_search


class PdfSearchTests(unittest.TestCase):
    def test_returns_hybrid_hits_without_topic_specific_reordering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "data/index/qdrant/active_structure.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"model": "embed-model"}))
            hits = [{"id": "first"}, {"id": "second"}]
            with patch("src.answer_agent.tools.filings.ROOT", root), \
                 patch("src.answer_agent.tools.filings.FireworksEmbedder", return_value=object()), \
                 patch("src.answer_agent.tools.filings.search_hybrid",
                       return_value={"results": hits}) as search:
                for question in ("supply chain risks", "revenue drivers"):
                    self.assertEqual(_pdf_search(question, "structure")["results"], hits)
                self.assertEqual(search.call_count, 2)


if __name__ == "__main__":
    unittest.main()

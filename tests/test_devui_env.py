"""The developer UI should read a local key without exposing or overriding it."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from devui.server import load_local_fireworks_key


class DeveloperEnvironmentTests(unittest.TestCase):
    def test_loads_quoted_key_from_env_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('FIREWORKS_API_KEY="example-key"\n')
            with patch.dict(os.environ, {}, clear=True):
                load_local_fireworks_key(path)
                self.assertEqual(os.environ["FIREWORKS_API_KEY"], "example-key")

    def test_exported_key_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('FIREWORKS_API_KEY="file-key"\n')
            with patch.dict(os.environ, {"FIREWORKS_API_KEY": "exported-key"}):
                load_local_fireworks_key(path)
                self.assertEqual(os.environ["FIREWORKS_API_KEY"], "exported-key")


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from src.mobalistic import clean_html, normalise, trusted_publisher, write_json, read_json

class MobalisticCoreTests(unittest.TestCase):
    def test_normalise_collapses_whitespace_and_case(self):
        self.assertEqual(normalise("  ONE   Esports "), "one esports")

    def test_clean_html_removes_tags(self):
        self.assertEqual(clean_html("<p>Patch <b>update</b></p>"), "Patch update")

    def test_trusted_publisher_uses_allowlist(self):
        self.assertTrue(trusted_publisher("ONE Esports - Mobile Legends", ["one esports"]))
        self.assertFalse(trusted_publisher("Random clickbait site", ["one esports"]))

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            write_json(path, {"published": ["url"]})
            self.assertEqual(read_json(path, {}), {"published": ["url"]})

    def test_build_registry_starts_empty_to_prevent_fake_builds(self):
        from src.mobalistic import BUILDS_PATH
        data = json.loads(BUILDS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data.get("builds"), [])

if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from src.mobalistic import clean_html, normalise, trusted_publisher, write_json, read_json, telegram_send_photo
import src.mobalistic as agent

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

    def test_memes_are_removed_from_editorial_config_and_engine(self):
        from src.mobalistic import FEEDS_PATH
        config = json.loads(FEEDS_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("memes", config.get("category_weights", {}))
        self.assertNotIn("memes", [item.get("category") for item in config.get("searches", [])])
        self.assertFalse(hasattr(agent, "make_meme_post"))

    def test_photo_and_caption_are_sent_in_one_telegram_api_call(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"ok": True, "result": {"message_id": 123}}
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "post.jpg"
            image.write_bytes(b"test-image")
            with patch.object(agent, "BOT_TOKEN", "123:test-token"), patch.object(agent, "CHANNEL_ID", "@mobalistic"), patch("src.mobalistic.requests.post", return_value=response) as post:
                telegram_send_photo(image, "<b>Заголовок</b>\nТекст поста")
        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertTrue(args[0].endswith("/sendPhoto"))
        self.assertEqual(kwargs["data"]["caption"], "<b>Заголовок</b>\nТекст поста")
        self.assertEqual(kwargs["data"]["parse_mode"], "HTML")


    def test_publication_is_capped_at_one_per_run(self):
        self.assertEqual(agent.MAX_POSTS_PER_RUN, 1)

    def test_editorial_image_is_not_generated_without_real_artwork(self):
        with patch.object(agent, "article_image", return_value=None):
            result = agent.create_editorial_card(
                {"brand":{"accent":"#A78BFA","background":"#0B0D16"}},
                "news", "Тестовый заголовок", "ONE Esports",
                source_url="https://example.com/article"
            )
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()

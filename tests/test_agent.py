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

    def test_build_registry_contains_only_complete_source_verified_builds(self):
        from src.mobalistic import BUILDS_PATH
        data = json.loads(BUILDS_PATH.read_text(encoding="utf-8"))
        builds = data.get("builds", [])
        self.assertGreaterEqual(len(builds), 1)
        for build in builds:
            self.assertTrue(build.get("verified"))
            self.assertTrue(build.get("source_url", "").startswith("https://"))
            self.assertTrue(build.get("patch_version"))
            self.assertEqual(len(build.get("items", [])), 6)
            self.assertTrue(all(item.get("name") and item.get("image_url", "").startswith("https://") for item in build["items"]))

    def test_builds_are_the_top_content_priority(self):
        from src.mobalistic import FEEDS_PATH
        config = json.loads(FEEDS_PATH.read_text(encoding="utf-8"))
        weights = config.get("category_weights", {})
        self.assertGreater(weights.get("builds", 0), max((v for k, v in weights.items() if k != "builds"), default=0))

    def test_hugging_face_paid_fallback_is_disabled_by_default(self):
        self.assertFalse(agent.HF_PAID_FALLBACK)

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



    def test_build_caption_is_clean_russian_and_keeps_item_order(self):
        build = {
            "hero": "Argus",
            "patch_version": "2.2.16a",
            "source_url": "https://example.com/argus",
            "context": "English source context must not leak into the Telegram caption",
            "items": [{"name": name, "image_url": "https://example.com/icon.png"} for name in [
                "Swift Boots", "Corrosion Scythe", "Demon Hunter Sword",
                "Golden Staff", "Haas's Claws", "Malefic Roar"
            ]],
            "emblem_name": "Marksman Emblem",
            "battle_spell_names": "Flicker",
        }
        caption = agent.build_caption(build)
        self.assertIn("Патч ", caption)
        self.assertIn("порядок закупки", caption)
        self.assertLess(caption.index("1. Swift Boots"), caption.index("2. Corrosion Scythe"))
        self.assertLess(caption.index("5. Haas&#x27;s Claws"), caption.index("6. Malefic Roar"))
        self.assertNotIn("English source context", caption)
        self.assertIn("Как играть:", caption)
        self.assertLessEqual(len(caption), 1024)

    def test_build_discovery_runs_even_when_registry_has_unposted_builds(self):
        source = agent.main.__code__
        import inspect
        code = inspect.getsource(agent.main)
        self.assertIn("api_build = discover_one_source_build(state)", code)
        self.assertNotIn("discover_one_source_build(state) if not builds", code)

    def test_build_caption_is_clean_russian_and_keeps_item_order(self):
        build = {
            "hero": "Argus",
            "patch_version": "2.2.16a",
            "source_url": "https://example.com/argus",
            "context": "This source context should not be copied into the caption",
            "items": [{"name": name, "image_url": "https://example.com/icon.png"} for name in [
                "Swift Boots", "Corrosion Scythe", "Demon Hunter Sword",
                "Golden Staff", "Haas's Claws", "Malefic Roar"
            ]],
            "emblem_name": "Marksman Emblem",
            "battle_spell_names": "Flicker",
        }
        caption = agent.build_caption(build)
        self.assertIn("Патч ", caption)
        self.assertIn("порядок закупки", caption)
        self.assertLess(caption.index("1. Swift Boots"), caption.index("2. Corrosion Scythe"))
        self.assertLess(caption.index("5. Haas&#x27;s Claws"), caption.index("6. Malefic Roar"))
        self.assertNotIn("This source context", caption)
        self.assertIn("Как играть:", caption)
        self.assertLessEqual(len(caption), 1024)

    def test_build_discovery_runs_even_when_registry_has_unposted_builds(self):
        import inspect
        code = inspect.getsource(agent.main)
        self.assertIn("api_build = discover_one_source_build(state)", code)
        self.assertNotIn("discover_one_source_build(state) if not builds", code)

    def test_publication_is_capped_at_one_per_run(self):
        self.assertEqual(agent.MAX_POSTS_PER_RUN, 1)

    def test_editorial_post_skips_when_no_publisher_or_ai_art_is_available(self):
        with patch.object(agent, "generate_ai_artwork", return_value=None), patch.object(agent, "article_image", return_value=None) as source_art:
            result = agent.create_editorial_card(
                {"brand":{"accent":"#A78BFA","background":"#0B0D16"}},
                "news", "Тестовый заголовок", "ONE Esports",
                source_url="https://example.com/article",
                image_url="https://example.com/source-image.jpg",
            )
        self.assertIsNone(result)
        source_art.assert_called()


    def test_google_news_artwork_is_rejected(self):
        response = Mock()
        response.url = "https://news.google.com/rss/articles/example"
        response.headers = {"Content-Type": "text/html"}
        response.raise_for_status.return_value = None
        with patch("src.mobalistic.requests.get", return_value=response):
            self.assertIsNone(agent.article_image("https://news.google.com/rss/articles/example"))

if __name__ == "__main__":
    unittest.main()

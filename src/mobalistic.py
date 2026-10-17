from __future__ import annotations

import base64
import html
import json
import logging
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import feedparser
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DRAFTS = DATA / "drafts"
STATE_PATH = DATA / "published.json"
FEEDS_PATH = DATA / "feeds.json"
BUILDS_PATH = DATA / "hero_builds.json"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("mobalistic")

PUBLISH_ENABLED = os.getenv("PUBLISH_ENABLED", "false").lower() == "true"
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "").strip()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell").strip()  # optional paid provider fallback
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image").strip()
HF_PAID_FALLBACK = os.getenv("HF_PAID_FALLBACK", "false").lower() == "true"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
MAX_POSTS_PER_RUN = 1  # Hard cap: never publish a burst. Build-first publisher revision.
MAX_AGE_DAYS = max(1, min(14, int(os.getenv("MAX_AGE_DAYS", "7"))))
REQUEST_TIMEOUT = 25
MIN_POST_INTERVAL = timedelta(hours=4)

CATEGORY_LABELS = {
    "builds": "СБОРКА ГЕРОЯ",
    "guides": "ГАЙД И ИДЕЯ",
    "news": "НОВОСТИ MLBB",
    "esports": "КИБЕРСПОРТ",
    "hot_topic": "ТЕМА ДНЯ",
}

@dataclass
class Candidate:
    url: str
    title: str
    publisher: str
    summary: str
    category: str
    published_at: str = ""
    image_url: str = ""

def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def load_config() -> dict[str, Any]:
    cfg = read_json(FEEDS_PATH, {})
    if not cfg.get("searches") or not cfg.get("trusted_publishers"):
        raise RuntimeError("data/feeds.json must define searches and trusted_publishers")
    return cfg

def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip().lower()

def clean_html(text: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text or "", flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(text))).strip()

def entry_date(entry: Any) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None

def trusted_publisher(publisher: str, allowed: list[str]) -> bool:
    name = normalise(publisher)
    return bool(name) and any(normalise(item) in name for item in allowed)

def collect_candidates(cfg: dict[str, Any], published_urls: set[str]) -> list[Candidate]:
    candidates: dict[str, Candidate] = {}
    allowed = cfg["trusted_publishers"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)

    for search in cfg["searches"]:
        category = str(search.get("category", "news"))
        query = str(search.get("query", "")).strip()
        if not query:
            continue
        rss_url = (
            "https://news.google.com/rss/search?q="
            + quote(query + f" when:{MAX_AGE_DAYS}d")
            + "&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            response = requests.get(
                rss_url, timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "MobalisticBot/1.0 (+RSS reader)"},
            )
            response.raise_for_status()
            feed = feedparser.parse(response.content)
        except Exception as exc:
            log.warning("RSS fetch failed for category=%s: %s", category, exc)
            continue

        for entry in feed.entries[:30]:
            url = str(entry.get("link", "")).strip()
            title = clean_html(str(entry.get("title", "")))
            source = entry.get("source", {})
            publisher = clean_html(str(source.get("title", ""))) if isinstance(source, dict) else ""
            summary = clean_html(str(entry.get("summary", "")))[:2400]
            published = entry_date(entry)

            if not url.startswith("https://") or not title or not publisher:
                continue
            if url in published_urls:
                continue
            if not trusted_publisher(publisher, allowed):
                continue
            if published is None or published < cutoff or published > datetime.now(timezone.utc) + timedelta(hours=12):
                continue

            image_url = ""
            for media in entry.get("media_content", []) or []:
                candidate_url = str(media.get("url", "")).strip() if isinstance(media, dict) else ""
                if candidate_url.startswith("https://"):
                    image_url = candidate_url
                    break
            if not image_url:
                for link in entry.get("links", []) or []:
                    candidate_url = str(link.get("href", "")).strip() if isinstance(link, dict) else ""
                    if candidate_url.startswith("https://") and "image" in str(link.get("type", "")).lower():
                        image_url = candidate_url
                        break
            candidate = Candidate(
                url=url, title=title, publisher=publisher,
                summary=summary, category=category,
                published_at=published.isoformat(), image_url=image_url,
            )
            candidates[url] = candidate

    # Stable order: newest first, with exact URLs deduplicated.
    result = sorted(candidates.values(), key=lambda c: c.published_at, reverse=True)
    log.info("RSS discovery found %s eligible unique candidate(s) across %s searches.", len(result), len(cfg["searches"]))
    return result

def gemini_text(prompt: str) -> str:
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing")
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{quote(GEMINI_MODEL, safe='')}:generateContent"
    )
    response = requests.post(
        endpoint,
        params={"key": GEMINI_KEY},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.55,
                "maxOutputTokens": 650,
                "responseMimeType": "text/plain",
            },
        },
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    answer = "\n".join(str(p.get("text", "")) for p in parts).strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty response")
    return answer

def generate_ai_artwork(prompt: str) -> Image.Image | None:
    """Try Gemini image generation first, then optional Hugging Face provider."""
    # Use the existing Gemini key first so Hugging Face's paid provider is not
    # treated as the only path. Free-tier image access depends on Google's account/model quota.
    if GEMINI_KEY:
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_IMAGE_MODEL}:generateContent",
                params={"key": GEMINI_KEY},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
                },
                timeout=90,
            )
            if response.ok:
                payload = response.json()
                for candidate in payload.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        inline = part.get("inlineData") or part.get("inline_data") or {}
                        encoded = inline.get("data")
                        if encoded:
                            from io import BytesIO
                            raw = base64.b64decode(encoded)
                            image = Image.open(BytesIO(raw)).convert("RGB")
                            if image.width >= 512 and image.height >= 512:
                                log.info("AI artwork generated using Gemini image model %s", GEMINI_IMAGE_MODEL)
                                return image
                log.warning("Gemini image model returned no usable image; trying optional fallback.")
            else:
                # Log status only: never leak API keys or request URLs containing them.
                log.warning("Gemini image generation unavailable (HTTP %s); trying optional fallback.", response.status_code)
        except Exception as exc:
            log.warning("Gemini image generation request failed (%s); trying optional fallback.", type(exc).__name__)
    if HF_TOKEN and HF_PAID_FALLBACK:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(provider="fal-ai", api_key=HF_TOKEN, timeout=90)
            image = client.text_to_image(prompt=prompt, model=HF_IMAGE_MODEL).convert("RGB")
            if image.width >= 512 and image.height >= 512:
                log.info("AI artwork generated using Hugging Face Inference Providers (fal-ai), model %s", HF_IMAGE_MODEL)
                return image
            log.warning("Hugging Face image provider returned an unexpectedly small image.")
        except Exception as exc:
            log.warning("Hugging Face image generation unavailable (provider access/quota may be limited): %s", exc)
    elif HF_TOKEN and not HF_PAID_FALLBACK:
        log.info("Hugging Face paid fallback is disabled; no paid image request was made.")
    return None

def art_prompt(category: str, title: str, hero: str = ""):
    subject = f"Mobile Legends hero {hero}" if hero else f"Mobile Legends Bang Bang editorial topic: {title}"
    return (
        "Create premium esports key art for a Mobile Legends: Bang Bang community media channel. "
        f"Main subject: {subject}. "
        "Vertical 4:5 composition, cinematic AAA fantasy game art, powerful readable silhouette, "
        "dramatic rim lighting, volumetric haze, detailed armor and materials, rich dark atmosphere, "
        "electric violet and icy blue accents, high contrast, polished official-game-promo quality, "
        "professional art direction, dynamic camera, clear focal point, subtle particles. "
        "No words, no typography, no letters, no logos, no watermark, no UI, no collage, no poster layout."
    )


def make_caption(item: Candidate) -> str | None:
    prompt = f"""
Ты — редактор живого русскоязычного Telegram-канала для игроков Mobile Legends: Bang Bang.
Пиши как игрок из комьюнити, не как пресс-релиз и не как нейросеть. Все имена героев пиши только официальным английским именем (Argus, Miya, Chou), никогда не транслитерируй и не переводи имя героя на русский. Если материал не о самой MLBB, нет конкретного факта, это SEO-перепечатка или данных недостаточно — ответь ровно SKIP.
Не выдумывай факты, даты, цифры, патчи, сборки, результаты, цитаты или мнения.
Строгий формат:
<b>Короткий заголовок по сути</b>

1–2 коротких абзаца, всего 2–4 предложения. Сразу факт и польза/последствие для игрока. Разговорный русский, естественный сленг только к месту (катка, пик, нерф, баф), максимум 1 эмодзи. Не повторяй заголовок, не пиши вступлений и пустого вопроса в конце.
В конце: 🔗 <a href="{item.url}">Источник</a>
HTML: только <b>, <i>, <a>. Максимум 450 символов с HTML.
Рубрика: {CATEGORY_LABELS.get(item.category, "MLBB")}
Издатель: {item.publisher}
Заголовок источника: {item.title}
Описание: {item.summary[:900]}
Ссылка: {item.url}
"""
    caption = gemini_text(prompt).strip()
    if caption.upper() == "SKIP" or len(clean_html(caption)) < 65:
        return None
    plain = normalise(clean_html(caption))
    low_signal = (
        "в мире mobile legends", "стало известно", "игроки смогут",
        "это событие подчеркивает", "данный шаг", "поклонники игры",
        "следите за обновлениями", "что думаете об этом"
    )
    if any(phrase in plain for phrase in low_signal):
        return None
    if len(clean_html(caption)) > 450 or len(caption) > 1024:
        return None
    if item.url not in caption:
        caption = caption.rstrip() + f'\n\n🔗 <a href="{html.escape(item.url, quote=True)}">Источник</a>'
    return caption if len(caption) <= 1024 else None

def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    options = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in options:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()

def wrap_by_pixels(draw: ImageDraw.ImageDraw, text: str, used_font: Any, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if draw.textbbox((0, 0), trial, font=used_font)[2] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def draw_background(cfg: dict[str, Any]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    width, height = 1080, 1350
    brand = cfg.get("brand", {})
    base = brand.get("background", "#0B0D16")
    accent = brand.get("accent", "#A78BFA")
    image = Image.new("RGB", (width, height), base)
    px = image.load()
    # Subtle top-left violet glow, calculated locally (no image API).
    ar, ag, ab = tuple(int(accent[i:i+2], 16) for i in (1, 3, 5))
    br, bg, bb = tuple(int(base[i:i+2], 16) for i in (1, 3, 5))
    for y in range(height):
        for x in range(width):
            distance = ((x + 80) ** 2 + (y - 40) ** 2) ** 0.5
            amount = max(0.0, 0.20 * (1 - distance / 1050))
            px[x, y] = (
                int(br * (1 - amount) + ar * amount),
                int(bg * (1 - amount) + ag * amount),
                int(bb * (1 - amount) + ab * amount),
            )
    draw = ImageDraw.Draw(image, "RGBA")
    # Editorial grid, frame, and accent details.
    for x in range(60, width, 72):
        draw.line((x, 0, x, height), fill=(255, 255, 255, 10), width=1)
    for y in range(60, height, 72):
        draw.line((0, y, width, y), fill=(255, 255, 255, 8), width=1)
    draw.rounded_rectangle((34, 34, width - 34, height - 34), radius=34, outline=(255, 255, 255, 38), width=2)
    draw.rounded_rectangle((62, 62, 76, 110), radius=7, fill=accent)
    draw.text((96, 65), brand.get("name", "MOBALISTIC"), font=font(29, True), fill="#FFFFFF")
    draw.text((96, 103), brand.get("tagline", "ИГРАЙ УМНЕЕ"), font=font(15, True), fill=accent)
    draw.text((70, 1272), "MOBALISTIC  •  MOBILE LEGENDS: BANG BANG", font=font(16, True), fill=(230, 232, 255, 180))
    return image, draw

def article_image(url: str) -> Image.Image | None:
    """Fetch a publisher's own article image when available; never invent a fake thumbnail."""
    if not url.startswith("https://"):
        return None
    try:
        response = requests.get(
            url, timeout=12,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MobalisticBot/1.0)"},
            allow_redirects=True,
        )
        response.raise_for_status()
        final_host = (urlparse(response.url).hostname or "").lower()
        if final_host == "news.google.com" or final_host.endswith(".google.com") or final_host == "google.com":
            log.info("Skipping Google-hosted placeholder artwork for %s", url)
            return None
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type.startswith("image/"):
            payload = response.content
        elif "text/html" in content_type or not content_type:
            html_text = response.text[:1_500_000]
            matches = re.findall(
                r'<meta[^>]+(?:property|name)=["\'](?:og:image|twitter:image)["\'][^>]+content=["\']([^"\']+)["\']|'
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\'](?:og:image|twitter:image)["\']',
                html_text, flags=re.I,
            )
            image_url = next((first or second for first, second in matches if (first or second).startswith("https://")), "")
            image_host = (urlparse(image_url).hostname or "").lower()
            if not image_url or image_host == "news.google.com" or image_host.endswith(".google.com") or image_host == "google.com":
                return None
            image_response = requests.get(
                image_url, timeout=12,
                headers={"User-Agent": "Mozilla/5.0 (compatible; MobalisticBot/1.0)"},
            )
            image_response.raise_for_status()
            if not image_response.headers.get("Content-Type", "").lower().startswith("image/"):
                return None
            payload = image_response.content
        else:
            return None
        if not payload or len(payload) > 5_000_000:
            return None
        from io import BytesIO
        return Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:
        log.info("Article artwork unavailable; use branded fallback: %s", exc)
        return None

def create_editorial_card(cfg: dict[str, Any], category: str, title: str, publisher: str = "", source_url: str = "", image_url: str = "", hero: str = "") -> Path:
    """Create an editorial image with real publisher artwork when available."""
    DRAFTS.mkdir(parents=True, exist_ok=True)
    width, height = 1080, 1350
    brand = cfg.get("brand", {})
    accent = brand.get("accent", "#A78BFA")
    base = brand.get("background", "#0B0D16")
    # Editorial art must be original AI artwork. Do not repost open-web/source images as a visual substitute.
    artwork = generate_ai_artwork(art_prompt(category, title, hero=hero))
    if artwork is None:
        log.info("Skipping editorial item: original AI artwork is unavailable; source images are intentionally not reused: %s", title)
        return None
    if artwork is not None:
        scale = max(width / artwork.width, height / artwork.height)
        artwork = artwork.resize((int(artwork.width * scale), int(artwork.height * scale)), Image.Resampling.LANCZOS)
        left, top = (artwork.width - width) // 2, (artwork.height - height) // 2
        image = artwork.crop((left, top, left + width, top + height)).filter(ImageFilter.GaussianBlur(1.5))
        shade = Image.new("RGBA", (width, height), (5, 7, 16, 112))
        image = Image.alpha_composite(image.convert("RGBA"), shade)
    else:
        image = Image.new("RGB", (width, height), base).convert("RGBA")
        glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse((-260, -250, 780, 780), fill=(167, 139, 250, 58))
        gd.ellipse((520, 650, 1450, 1550), fill=(65, 95, 255, 28))
        image = Image.alpha_composite(image, glow.filter(ImageFilter.GaussianBlur(110)))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, width, 14), fill=accent)
    draw.rounded_rectangle((54, 54, width - 54, height - 54), radius=30, outline=(255, 255, 255, 48), width=2)
    draw.text((76, 74), "MOBALISTIC", font=font(28, True), fill=(255, 255, 255, 255))
    draw.text((78, 116), CATEGORY_LABELS.get(category, "MLBB").upper(), font=font(18, True), fill=accent)
    draw.line((76, 164, 270, 164), fill=accent, width=5)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for y in range(480, 1190):
        alpha = int(20 + 205 * ((y - 480) / 710))
        od.line((0, y, width, y), fill=(5, 7, 16, alpha))
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = font(64, True)
    lines = wrap_by_pixels(draw, title, title_font, 900)
    if len(lines) > 5:
        title_font = font(52, True)
        lines = wrap_by_pixels(draw, title, title_font, 900)
    y = min(800, max(430, 1110 - len(lines) * (title_font.size + 15)))
    for line in lines[:6]:
        draw.text((76, y), line, font=title_font, fill="#FFFFFF", stroke_width=1, stroke_fill=(0, 0, 0, 80))
        y += title_font.size + 17
    draw.text((78, 1175), (publisher or "MOBILE LEGENDS: BANG BANG")[:70], font=font(18, True), fill=(235, 235, 250, 225))
    draw.text((78, 1220), "ИГРАЙ УМНЕЕ  /  MOBALISTIC", font=font(16, True), fill=accent)
    out = DRAFTS / f"card-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}.jpg"
    image.convert("RGB").save(out, "JPEG", quality=91, optimize=True)
    return out

def source_text_value(value: Any) -> str:
    """Extract a human-readable label only from values explicitly present in the source API."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("name", "title", "label", "display_name"):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key].strip()
    if isinstance(value, list):
        labels = [source_text_value(item) for item in value]
        return ", ".join(label for label in labels if label)
    return ""


def source_image_value(value: Any) -> str:
    if isinstance(value, str) and value.startswith("https://"):
        return value
    if isinstance(value, dict):
        for key in ("image_url", "icon_url", "image", "icon", "url", "src"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.startswith("https://"):
                return nested
    if isinstance(value, list):
        for item in value:
            found = source_image_value(item)
            if found:
                return found
    return ""


def image_url_from_hero(data: dict[str, Any]) -> str:
    """Return a hero artwork URL only when the upstream API explicitly provides one."""
    keys = ("splash_art", "splashArt", "artwork_url", "artwork", "portrait_url", "portraitUrl", "portrait", "image_url", "imageUrl", "image", "icon_url", "iconUrl", "icon", "avatar", "cover_url", "cover")
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.startswith("https://"):
            return value
        if isinstance(value, dict):
            for nested_key in ("url", "src", "large", "original"):
                nested = value.get(nested_key)
                if isinstance(nested, str) and nested.startswith("https://"):
                    return nested
    return ""

def create_build_card(cfg: dict[str, Any], build: dict[str, Any]) -> Path | None:
    """Minimal build artwork: hero splash on top, six ordered item icons below, no decorative frames."""
    required = ("hero", "patch_version", "source_url", "verified_at", "items")
    if not build.get("verified") or any(not build.get(key) for key in required):
        log.warning("Skipping unverified/incomplete build: %s", build.get("hero", "unknown"))
        return None
    items = build["items"]
    if len(items) != 6 or any(not x.get("name") or not str(x.get("image_url", "")).startswith("https://") for x in items):
        log.warning("Build needs exactly six named items with HTTPS icons: %s", build.get("hero"))
        return None
    if not str(build["source_url"]).startswith("https://"):
        return None

    from io import BytesIO
    icons: list[Image.Image] = []
    for item in items:
        try:
            response = requests.get(item["image_url"], timeout=REQUEST_TIMEOUT,
                                    headers={"User-Agent": "MobalisticBot/1.0"})
            response.raise_for_status()
            if len(response.content) > 3_000_000:
                return None
            icon = Image.open(BytesIO(response.content)).convert("RGBA")
            if icon.width < 24 or icon.height < 24:
                return None
            icons.append(icon)
        except Exception as exc:
            log.warning("Cannot load icon for %s: %s", item.get("name"), exc)
            return None

    # Prefer the actual hero splash supplied by the build source. AI art is only a fallback,
    # never a reason to replace the correct item order or use an unrelated web image.
    hero_art = None
    hero_url = str(build.get("hero_image_url", ""))
    if hero_url.startswith("https://"):
        hero_art = article_image(hero_url)
    if hero_art is None:
        hero_art = generate_ai_artwork(art_prompt("builds", f"{build['hero']} hero", str(build["hero"])))
    if hero_art is None:
        log.warning("No usable hero splash or generated art for %s; skip rather than publish a broken card", build["hero"])
        return None

    width, height = 1080, 1350
    hero_height = 900
    scale = max(width / hero_art.width, hero_height / hero_art.height)
    hero_art = hero_art.resize((int(hero_art.width * scale), int(hero_art.height * scale)), Image.Resampling.LANCZOS)
    left = (hero_art.width - width) // 2
    top = max(0, min(hero_art.height - hero_height, int((hero_art.height - hero_height) * 0.35)))
    hero_crop = hero_art.crop((left, top, left + width, top + hero_height)).convert("RGBA")
    image = Image.new("RGBA", (width, height), (10, 12, 20, 255))
    image.alpha_composite(hero_crop, (0, 0))

    # A simple dark fade lets the item row read clearly while keeping the hero artwork dominant.
    fade = Image.new("RGBA", (width, hero_height), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fade)
    for y in range(560, hero_height):
        alpha = int(8 + 110 * ((y - 560) / (hero_height - 560)))
        fd.line((0, y, width, y), fill=(5, 7, 14, alpha))
    image.alpha_composite(fade, (0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((48, 42), str(build["hero"]).upper(), font=font(54, True), fill=(255, 255, 255, 245),
              stroke_width=2, stroke_fill=(0, 0, 0, 150))
    draw.text((52, 108), f"СБОРКА · ПАТЧ {build['patch_version']}", font=font(22, True),
              fill=(235, 231, 255, 240), stroke_width=1, stroke_fill=(0, 0, 0, 120))

    # Flat, borderless item strip: six icons in the exact source order.
    strip_y = hero_height
    draw.rectangle((0, strip_y, width, height), fill=(10, 12, 20, 255))
    margin = 34
    cell_w = (width - 2 * margin) // 6
    icon_size = 116
    for idx, (item, icon) in enumerate(zip(items, icons)):
        x = margin + idx * cell_w + (cell_w - icon_size) // 2
        y = strip_y + 40
        icon.thumbnail((icon_size, icon_size), Image.Resampling.LANCZOS)
        image.alpha_composite(icon, (x, y))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.text((margin + idx * cell_w + 4, strip_y + 19), str(idx + 1),
                  font=font(18, True), fill=(180, 158, 255, 255))
        item_font = font(15, True)
        lines = wrap_by_pixels(draw, str(item["name"]), item_font, cell_w - 8)
        text_y = strip_y + 170
        for line in lines[:3]:
            box = draw.textbbox((0, 0), line, font=item_font)
            draw.text((margin + idx * cell_w + (cell_w - (box[2] - box[0])) // 2, text_y),
                      line, font=item_font, fill=(245, 245, 250, 255))
            text_y += 20

    out = DRAFTS / f"build-{re.sub(r'[^a-zA-Z0-9_-]+', '-', str(build['hero']).lower())}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jpg"
    image.convert("RGB").save(out, "JPEG", quality=94, optimize=True)
    log.info("Created clean hero build card: hero=%s items=6 patch=%s", build["hero"], build["patch_version"])
    return out

def create_hero_build_poster(cfg: dict[str, Any], hero: str, hero_image_url: str, build_label: str,
                             emblem_text: str, talent_text: str, spell_text: str) -> Path | None:
    """Create a polished editorial meta card from real splash artwork; no AI image service needed."""
    # Ask the image model for original key art first; retain source artwork as fallback.
    artwork = generate_ai_artwork(art_prompt("builds", f"{hero} hero build", hero))
    if artwork is None and hero_image_url:
        artwork = article_image(hero_image_url)
    if artwork is None:
        log.warning("No AI or genuine splash artwork available for %s; skip hero build poster", hero)
        return None
    width, height = 1080, 1350
    brand = cfg.get("brand", {})
    accent_hex = brand.get("accent", "#A78BFA")
    accent = tuple(int(accent_hex[i:i+2], 16) for i in (1, 3, 5))
    scale = max(width / artwork.width, height / artwork.height)
    artwork = artwork.resize((int(artwork.width * scale), int(artwork.height * scale)), Image.Resampling.LANCZOS)
    left, top = (artwork.width - width) // 2, (artwork.height - height) // 2
    art = artwork.crop((left, top, left + width, top + height)).convert("RGBA")
    art = Image.alpha_composite(art, Image.new("RGBA", (width, height), (3, 5, 14, 42)))
    fade = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fade)
    for y in range(390, height):
        t = (y - 390) / (height - 390)
        fd.line((0, y, width, y), fill=(5, 7, 17, int(55 + 190 * t)))
    art = Image.alpha_composite(art, fade)
    draw = ImageDraw.Draw(art, "RGBA")
    draw.rounded_rectangle((52, 48, 360, 112), radius=25, fill=(8, 10, 22, 205), outline=(255,255,255,70), width=2)
    draw.rounded_rectangle((70, 66, 80, 94), radius=5, fill=accent + (255,))
    draw.text((100, 63), "MOBALISTIC", font=font(25, True), fill="#FFFFFF")
    draw.rounded_rectangle((770, 58, 1028, 108), radius=20, fill=accent + (240,))
    draw.text((795, 72), "COMMUNITY META", font=font(16, True), fill="#11121B")
    draw.text((58, 720), "HERO BUILD  /  #01", font=font(21, True), fill=accent + (255,))
    draw.text((52, 755), hero.upper(), font=font(94, True), fill="#FFFFFF", stroke_width=2, stroke_fill=(0,0,0,100))
    draw.text((60, 865), "WHAT PLAYERS PICK MOST", font=font(21, True), fill=(235,238,255,245))
    draw.line((60, 908, 1020, 908), fill=(255,255,255,90), width=2)
    rows = [
        ("EMBLEM", emblem_text, 21 / 43, "49%"),
        ("TALENT", talent_text, 27 / 43, "63%"),
        ("BATTLE SPELL", spell_text, 0.49, "49%"),
    ]
    y = 938
    for label, value, ratio, pct in rows:
        draw.rounded_rectangle((56, y, 1024, y + 92), radius=20, fill=(10, 12, 27, 205), outline=(255,255,255,48), width=2)
        draw.text((82, y + 13), label, font=font(15, True), fill=accent + (255,))
        draw.text((82, y + 40), value, font=font(25, True), fill="#FFFFFF")
        bx1, by1, bx2, by2 = 770, y + 54, 930, y + 67
        draw.rounded_rectangle((bx1, by1, bx2, by2), radius=6, fill=(255,255,255,38))
        draw.rounded_rectangle((bx1, by1, bx1 + int((bx2-bx1)*ratio), by2), radius=6, fill=accent + (255,))
        draw.text((944, y + 45), pct, font=font(18, True), fill="#FFFFFF")
        y += 101
    draw.rounded_rectangle((58, 1262, 1022, 1318), radius=18, fill=(8,10,22,220), outline=(255,255,255,38), width=1)
    draw.text((80, 1278), "PICK RATE IS NOT WIN RATE  /  ADAPT TO MATCHUP", font=font(15, True), fill=(230,232,245,245))
    DRAFTS.mkdir(parents=True, exist_ok=True)
    out = DRAFTS / f"hero-build-{re.sub(r'[^a-zA-Z0-9_-]+', '-', hero.lower())}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}.jpg"
    art.convert("RGB").save(out, "JPEG", quality=94, optimize=True)
    return out

def telegram_send_photo(image_path: Path, caption: str) -> None:
    """Publish artwork and its readable HTML text as ONE Telegram photo post."""
    if not BOT_TOKEN or not CHANNEL_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID are required to publish")
    endpoint = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with image_path.open("rb") as photo:
        response = requests.post(
            endpoint,
            data={"chat_id": CHANNEL_ID, "caption": caption[:1024], "parse_mode": "HTML"},
            files={"photo": (image_path.name, photo, "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png")},
            timeout=45,
        )
    try:
        result = response.json()
    except ValueError:
        response.raise_for_status()
        raise RuntimeError("Telegram sendPhoto returned a non-JSON response")
    if response.status_code >= 400 or not result.get("ok"):
        description = str(result.get("description", response.text))[:500]
        raise RuntimeError(f"Telegram sendPhoto failed (HTTP {response.status_code}): {description}")

def record_preview(image_path: Path, caption: str, category: str) -> None:
    preview_path = image_path.with_suffix(".txt")
    preview_path.write_text(
        f"CATEGORY: {category}\nPUBLISH_ENABLED: false\nTELEGRAM_LAYOUT: one photo post with readable HTML caption\n\nPOST TEXT:\n{caption}\n\nIMAGE FILE: {image_path.name}\n",
        encoding="utf-8",
    )
    log.info("PREVIEW ONLY — category=%s image=%s post_text=%s", category, image_path, caption.replace("\n", " | "))

MLBBDEX_API = "https://mlbbdex.com/api/v1"
MLBBDEX_SITE = "https://mlbbdex.com"

def mlbbdex_get(path: str) -> dict[str, Any]:
    response = requests.get(
        f"{MLBBDEX_API}/{path.lstrip('/')}",
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "MobalisticBot/1.0 (+https://mlbbdex.com/api-doc)"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError(f"Unexpected MLBBDex API response for {path}")
    return payload

def canonical_item_name(value: str) -> str:
    name = normalise(value)
    name = re.sub(r"^(?:i\s+)?ice hunter's\s+", "", name)
    name = re.sub(r"^(?:a\s+)?", "", name)
    # Blessing variants share the base equipment icon/name.
    name = name.split(" - ", 1)[0].strip()
    return name

def discover_one_source_build(state: dict[str, Any]) -> dict[str, Any] | None:
    """Discover a fresh source-backed build from the live API, rotating across heroes."""
    try:
        heroes_payload = mlbbdex_get("heroes")
        items_payload = mlbbdex_get("items")
        patches_payload = mlbbdex_get("patches")
        heroes = heroes_payload.get("data", [])
        items = items_payload.get("data", [])
        patches = patches_payload.get("data", [])
        if not heroes or not items or not patches:
            log.warning("MLBBDex returned incomplete hero/item/patch data")
            return None

        item_by_name = {
            canonical_item_name(str(item.get("name", ""))): item
            for item in items if isinstance(item, dict) and item.get("name") and item.get("slug")
        }
        patch = patches[0] if isinstance(patches[0], dict) else {}
        patch_version = next(
            (str(patch.get(key)).strip() for key in ("version", "name", "patch", "number", "slug")
             if patch.get(key) not in (None, "")),
            "",
        )
        if not patch_version:
            log.warning("Cannot determine current patch from MLBBDex; skip build rather than guess")
            return None

        heroes = [h for h in heroes if isinstance(h, dict) and h.get("slug") and h.get("name")]
        if not heroes:
            return None
        heroes.sort(key=lambda h: (normalise(str(h["name"])) != "argus", normalise(str(h["name"]))))
        cursor = int(state.get("build_cursor", 0)) % len(heroes)
        scan_count = min(8, len(heroes))
        published = set(state.get("published", []))

        for offset in range(scan_count):
            hero_index = (cursor + offset) % len(heroes)
            hero = heroes[hero_index]
            try:
                detail_payload = mlbbdex_get(f"heroes/{quote(str(hero['slug']))}")
            except Exception as exc:
                log.warning("Could not load build data for %s: %s", hero["name"], exc)
                continue
            detail = detail_payload.get("data", {})
            source_builds = detail.get("builds", [])
            if not isinstance(source_builds, list) or not source_builds:
                continue

            for source_build in source_builds:
                if not isinstance(source_build, dict):
                    continue
                names = source_build.get("items", [])
                if not isinstance(names, list) or len(names) != 6:
                    continue
                resolved = []
                for raw_name in names:
                    item = item_by_name.get(canonical_item_name(str(raw_name)))
                    if not item:
                        resolved = []
                        break
                    icon_url = image_url_from_hero(item)
                    if not icon_url:
                        resolved = []
                        break
                    resolved.append({"name": str(item["name"]), "image_url": icon_url})
                if len(resolved) != 6:
                    log.info("Skipping %s build: catalog did not resolve all six item icons", hero["name"])
                    continue

                source_url = f"{MLBBDEX_SITE}/en/heroes/{hero['slug']}"
                build_name = str(source_build.get("name") or "Build")
                build = {
                    "hero": str(detail.get("name") or hero["name"]),
                    "hero_image_url": image_url_from_hero(detail) or image_url_from_hero(hero),
                    "patch_version": patch_version,
                    "source_url": source_url,
                    "verified_at": datetime.now(timezone.utc).date().isoformat(),
                    "verified": True,
                    "verification_method": "MLBBDex live API + exact six-item catalog match",
                    "source_build_name": build_name,
                    "context": "",
                    "emblem_name": next(
                        (source_text_value(source_build.get(k)) for k in ("emblem", "emblem_name", "emblem_build", "emblems")
                         if source_text_value(source_build.get(k))), ""
                    ),
                    "emblem_image_url": next(
                        (source_image_value(source_build.get(k)) for k in ("emblem", "emblem_build", "emblems")
                         if source_image_value(source_build.get(k))), ""
                    ),
                    "battle_spell_names": next(
                        (source_text_value(source_build.get(k)) for k in ("battle_spells", "battle_spell", "spells", "spell")
                         if source_text_value(source_build.get(k))), ""
                    ),
                    "battle_spell_image_urls": [
                        source_image_value(source_build.get(k))
                        for k in ("battle_spells", "battle_spell", "spells", "spell")
                        if source_image_value(source_build.get(k))
                    ],
                    "items": resolved,
                    "attribution": "Build data from MLBBDex; item and patch details are linked to the source.",
                }
                key = f"build:{build['hero']}:{patch_version}:{build_name}"
                if key in published or f"build:{build['hero']}:{patch_version}" in published:
                    continue
                state["build_cursor"] = hero_index
                log.info("Live build discovery found %s / %s / patch %s", build["hero"], build_name, patch_version)
                return build

        state["build_cursor"] = (cursor + scan_count) % len(heroes)
        log.info("No unpublished verified build found across %s heroes this run.", scan_count)
        return None
    except Exception as exc:
        log.warning("MLBBDex live build discovery unavailable; checking reviewed fallback registry: %s", exc)
        return None

def eligible_builds() -> list[dict[str, Any]]:
    data = read_json(BUILDS_PATH, {"builds": []})
    return [
        build for build in data.get("builds", [])
        if build.get("verified") is True
        and len(build.get("items", [])) == 6
        and build.get("source_url")
        and build.get("patch_version")
        and build.get("verified_at")
    ]

def build_caption(build: dict[str, Any]) -> str:
    items = "\n".join(f"{i}. {item['name']}" for i, item in enumerate(build["items"], 1))
    lines = [
        f"⚔️ <b>{html.escape(str(build['hero']))} — сборка</b>",
        f"Версия игры: <b>{html.escape(str(build['patch_version']))}</b>",
        "",
        "<b>Предметы по порядку</b>",
        items,
    ]
    if build.get("emblem_name"):
        lines.extend(["", f"🧬 Эмблема: <b>{html.escape(str(build['emblem_name']))}</b>"])
    if build.get("battle_spell_names"):
        lines.append(f"⚡ Заклинание: <b>{html.escape(str(build['battle_spell_names']))}</b>")
    lines.extend([
        "",
        "Это рекомендация источника, а не универсальная сборка: подстраивай предметы под состав соперника.",
        f"🔎 <a href=\"{html.escape(str(build['source_url']), quote=True)}\">Источник сборки</a>",
        "",
        "#MLBB #MobileLegends #Mobalistic",
    ])
    caption = "\n".join(lines)
    if len(caption) > 1024:
        raise ValueError("Build caption exceeds Telegram limit")
    return caption

def select_category(cfg: dict[str, Any], available: set[str], counts: dict[str, int]) -> str | None:
    weights = cfg.get("category_weights", {})
    # Only select a category when there is real material for it.
    choices = [(name, int(weight)) for name, weight in weights.items() if name in available and int(weight) > 0]
    if not choices:
        return None
    # Prefer categories below their target share to improve the long-term mix.
    total = max(1, sum(counts.values()))
    def score(pair: tuple[str, int]) -> float:
        name, weight = pair
        actual = counts.get(name, 0) / total
        target = weight / max(1, sum(int(v) for v in weights.values()))
        return (target - actual) * 1000 + weight
    return max(choices, key=score)[0]

def main() -> int:
    cfg = load_config()
    DRAFTS.mkdir(parents=True, exist_ok=True)
    state = read_json(STATE_PATH, {"published": [], "last_category": None, "category_counts": {}})
    published = set(state.get("published", []))
    counts = state.get("category_counts", {})

    if PUBLISH_ENABLED and (not BOT_TOKEN or not CHANNEL_ID or not GEMINI_KEY):
        log.error("Publishing is enabled but required GitHub secrets are missing.")
        return 2
    if not GEMINI_KEY:
        log.error("GEMINI_API_KEY is missing. Add it to GitHub Actions secrets before running the content engine.")
        return 2

    builds = [
        b for b in eligible_builds()
        if f"build:{b.get('hero')}:{b.get('patch_version')}:{b.get('source_build_name', 'Build')}" not in published
        and f"build:{b.get('hero')}:{b.get('patch_version')}" not in published
    ]
    # The agent discovers source-backed builds on every run; the registry is only a fallback.
    api_build = discover_one_source_build(state)
    if api_build:
        builds = [api_build] + [
            b for b in builds
            if f"build:{b.get('hero')}:{b.get('patch_version')}:{b.get('source_build_name', 'Build')}" !=
               f"build:{api_build.get('hero')}:{api_build.get('patch_version')}:{api_build.get('source_build_name', 'Build')}"
        ]

    last_post_raw = str(state.get("last_post_at", "")).strip()
    if PUBLISH_ENABLED and last_post_raw and not api_build and not builds:
        try:
            last_post_at = datetime.fromisoformat(last_post_raw.replace("Z", "+00:00"))
            if last_post_at.tzinfo is None:
                last_post_at = last_post_at.replace(tzinfo=timezone.utc)
            wait_left = MIN_POST_INTERVAL - (datetime.now(timezone.utc) - last_post_at.astimezone(timezone.utc))
            if wait_left.total_seconds() > 0:
                log.info("Cooldown active; next publication allowed in about %s minutes.", int(wait_left.total_seconds() // 60))
                return 0
        except ValueError:
            log.warning("Invalid last_post_at in state; cooldown cannot be evaluated.")

    candidates = collect_candidates(cfg, published)
    available: set[str] = set()
    if builds:
        available.add("builds")
    for candidate in candidates:
        if candidate.category in {"guides", "news", "esports", "hot_topic"}:
            available.add(candidate.category)
    if not candidates and not builds:
        log.info("No eligible news or verified build found; skip this run.")
    
    posted = 0
    used_urls: set[str] = set()
    while posted < 1:  # One post maximum per run.
        remaining = [c for c in candidates if c.url not in used_urls]
        choices: set[str] = set()
        if builds:
            choices.add("builds")
        choices.update(c.category for c in remaining if c.category in {"guides", "news", "esports", "hot_topic"})
        category = select_category(cfg, choices, counts)
        if category is None:
            break

        image_path: Path | None = None
        caption = ""
        source_key = ""
        if category == "builds":
            selected = next((
                b for b in builds
                if f"build:{b['hero']}:{b['patch_version']}:{b.get('source_build_name', 'Build')}" not in published
                and f"build:{b['hero']}:{b['patch_version']}" not in published
            ), None)
            if selected is None:
                builds = []
                continue
            image_path = create_build_card(cfg, selected)
            if image_path is None:
                builds = [b for b in builds if b is not selected]
                continue
            caption = build_caption(selected)
            source_key = f"build:{selected['hero']}:{selected['patch_version']}:{selected.get('source_build_name', 'Build')}"
            builds = [b for b in builds if b is not selected]
        else:
            selected_candidate = next((c for c in remaining if c.category == category), None)
            if selected_candidate is None:
                continue
            try:
                caption = make_caption(selected_candidate)
            except Exception as exc:
                log.warning("AI caption generation failed; skip source %s: %s", selected_candidate.url, exc)
                used_urls.add(selected_candidate.url)
                continue
            used_urls.add(selected_candidate.url)
            if not caption:
                continue
            image_title = re.sub(r"<[^>]+>", "", caption.splitlines()[0]).strip()[:110]
            image_title = html.unescape(image_title) or selected_candidate.title
            image_path = create_editorial_card(
                cfg, category, image_title, selected_candidate.publisher,
                source_url=selected_candidate.url, image_url=selected_candidate.image_url,
            )
            source_key = selected_candidate.url

        if image_path is None or not caption:
            continue
        if PUBLISH_ENABLED:
            try:
                # One Telegram message: image plus readable caption, never two separate posts.
                telegram_send_photo(image_path, caption)
            except Exception:
                log.exception("Telegram publication failed; state was not advanced")
                return 3
            published.add(source_key)
            state["last_post_at"] = datetime.now(timezone.utc).isoformat()
            if category == "builds" and selected.get("verification_method"):
                state["build_cursor"] = int(state.get("build_cursor", 0)) + 1
            state["published"] = list(published)[-3000:]
            counts[category] = int(counts.get(category, 0)) + 1
            state["category_counts"] = counts
            state["last_category"] = category
            write_json(STATE_PATH, state)
            log.info("Published category=%s source=%s", category, source_key)
        else:
            record_preview(image_path, caption, category)
        posted += 1

    if posted == 0:
        log.info("No eligible new posts this run.")
    else:
        log.info("Run finished. %s post(s) %s.", posted, "published" if PUBLISH_ENABLED else "prepared for preview")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log.exception("Fatal agent error: %s", exc)
        sys.exit(1)

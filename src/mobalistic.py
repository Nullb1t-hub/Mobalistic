from __future__ import annotations

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
from PIL import Image, ImageDraw, ImageFont

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
MAX_POSTS_PER_RUN = max(1, min(5, int(os.getenv("MAX_POSTS_PER_RUN", "2"))))
MAX_AGE_DAYS = max(1, min(14, int(os.getenv("MAX_AGE_DAYS", "7"))))
REQUEST_TIMEOUT = 25

CATEGORY_LABELS = {
    "builds": "СБОРКА ГЕРОЯ",
    "guides": "ГАЙД И ИДЕЯ",
    "news": "НОВОСТИ MLBB",
    "esports": "КИБЕРСПОРТ",
    "memes": "MLBB MEME",
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

            candidate = Candidate(
                url=url, title=title, publisher=publisher,
                summary=summary, category=category,
                published_at=published.isoformat(),
            )
            candidates[url] = candidate

    # Stable order: newest first, with exact URLs deduplicated.
    return sorted(candidates.values(), key=lambda c: c.published_at, reverse=True)

def gemini_text(prompt: str) -> str:
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing")
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
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

def make_caption(item: Candidate) -> str | None:
    prompt = f"""
Ты — редактор канала MOBALISTIC о Mobile Legends: Bang Bang.
Пиши на русском естественно, как опытный игрок, без канцелярита и без фраз «как ИИ».
Не придумывай факты, числа, даты, результаты матчей, патчи, цитаты, статистику или предметы.
Используй только предоставленные ниже заголовок и краткое описание. Если фактов мало или материал не относится к MLBB, ответь ровно SKIP.
Сделай полезный пост: цепляющий короткий заголовок, 2–4 коротких абзаца, уместный игровой сленг, максимум 3 эмодзи.
Не утверждай, что сборка метовая или профессиональная, если источник этого не доказывает.
Не копируй описание дословно. В конце отдельной строкой укажи «Источник: {item.url}».
Максимум 850 символов до ссылки.

Рубрика: {CATEGORY_LABELS.get(item.category, "MLBB")}
Издатель: {item.publisher}
Дата публикации источника: {item.published_at}
Заголовок источника: {item.title}
Описание источника: {item.summary}
Ссылка: {item.url}
"""
    caption = gemini_text(prompt).strip()
    if caption.upper() == "SKIP" or len(caption) < 70:
        return None
    if item.url not in caption:
        caption = caption.rstrip() + f"\n\nИсточник: {item.url}"
    if len(caption) > 1000:
        caption = caption[:940].rsplit(" ", 1)[0] + f"…\n\nИсточник: {item.url}"
    return caption

def make_meme_post() -> tuple[str, str]:
    """Generate an original, evergreen gameplay meme without asserting news or game stats."""
    prompt = """
Придумай ОДИН оригинальный короткий мем для русскоязычного Telegram-канала MOBALISTIC про Mobile Legends: Bang Bang.
Тема — узнаваемая ситуация из обычного матча (лесник, соло-линия, тиммейты, ранкед, ласт пик и т.п.).
Не оскорбляй конкретных игроков, не используй ненависть, не выдумывай новости, патчи или реальные цитаты.
Стиль — живой игровой юмор, не похоже на рекламный текст.
Ответь строго в формате:
ЗАГОЛОВОК: короткая фраза до 65 символов
ТЕКСТ: две короткие строки, подходящие для подписи к карточке
"""
    raw = gemini_text(prompt).strip()
    title_match = re.search(r"ЗАГОЛОВОК:\s*(.+)", raw)
    text_match = re.search(r"ТЕКСТ:\s*([\s\S]+)", raw)
    if not title_match or not text_match:
        raise RuntimeError("Gemini meme response did not match the required format")
    title = title_match.group(1).strip()[:90]
    body = text_match.group(1).strip()
    if len(title) < 8 or len(body) < 15:
        raise RuntimeError("Gemini meme response was too short")
    caption = f"🎮 {title}\n\n{body}\n\n#MLBB #Mobalistic"
    return title, caption[:1000]

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

def create_editorial_card(cfg: dict[str, Any], category: str, title: str, publisher: str = "") -> Path:
    DRAFTS.mkdir(parents=True, exist_ok=True)
    image, draw = draw_background(cfg)
    accent = cfg.get("brand", {}).get("accent", "#A78BFA")
    label = CATEGORY_LABELS.get(category, "MLBB")
    draw.rounded_rectangle((70, 205, 360, 258), radius=24, fill=accent)
    draw.text((92, 218), label, font=font(19, True), fill="#10111A")

    # Large typographic composition with deliberate line breaks.
    title_font = font(66, True)
    lines = wrap_by_pixels(draw, title, title_font, 870)
    if len(lines) > 5:
        title_font = font(54, True)
        lines = wrap_by_pixels(draw, title, title_font, 870)
    y = 330
    for line in lines[:6]:
        draw.text((70, y), line, font=title_font, fill="#FFFFFF", stroke_width=0)
        y += title_font.size + 20

    # Decorative game-inspired geometric shapes, not copied game artwork.
    draw.rounded_rectangle((70, 870, 1010, 1120), radius=28, fill=(15, 17, 31, 230), outline=(167, 139, 250, 100), width=2)
    if category == "memes":
        panel_lines = ("КОГДА РАНКЕД ПОШЁЛ НЕ ТУДА", "МЫ ВСЕ ЭТО ПРОХОДИЛИ", "MOBALISTIC • ЮМОР ПО-НАШЕМУ")
    elif category == "news" or category == "esports" or category == "hot_topic":
        panel_lines = ("СОБЫТИЯ ИЗ МИРА MLBB", "КОНТЕКСТ ВАЖНЕЕ СЛУХОВ", "Читай источник и следи за обновлениями")
    elif category == "guides":
        panel_lines = ("МЕНЬШЕ ТИЛЬТА.", "БОЛЬШЕ ОСОЗНАННЫХ РЕШЕНИЙ.", "Сохрани идею и проверь в игре")
    else:
        panel_lines = ("МЕТА МЕНЯЕТСЯ.", "ИГРАЙ С ГОЛОВОЙ.", "Проверяй патч • адаптируйся • побеждай")
    draw.text((105, 920), panel_lines[0], font=font(27, True), fill="#FFFFFF")
    draw.text((105, 968), panel_lines[1], font=font(27, True), fill=accent)
    draw.text((105, 1034), panel_lines[2], font=font(19), fill=(225, 225, 240, 230))
    if publisher:
        source_label = publisher[:70]
        draw.text((70, 1170), "ИСТОЧНИК  /  " + source_label, font=font(17), fill=(220, 220, 238, 190))
    out = DRAFTS / f"card-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}.png"
    image.save(out, "PNG", optimize=True)
    return out

def create_build_card(cfg: dict[str, Any], build: dict[str, Any]) -> Path | None:
    # Fail closed: never render/publish an unverified or incomplete build.
    required = ("hero", "patch_version", "source_url", "verified_at", "items")
    if not build.get("verified") or any(not build.get(key) for key in required):
        log.warning("Skipping unverified/incomplete build: %s", build.get("hero", "unknown"))
        return None
    items = build["items"]
    if len(items) != 6 or any(not x.get("name") or not x.get("image_url") for x in items):
        log.warning("Build needs exactly six named items with image URLs: %s", build.get("hero"))
        return None
    if not str(build["source_url"]).startswith("https://"):
        return None

    # Download every item icon before drawing; if one is missing, do not fake it.
    icons: list[Image.Image] = []
    for item in items:
        try:
            response = requests.get(item["image_url"], timeout=REQUEST_TIMEOUT, headers={"User-Agent": "MobalisticBot/1.0"})
            response.raise_for_status()
            if len(response.content) > 3_000_000:
                return None
            from io import BytesIO
            icon = Image.open(BytesIO(response.content)).convert("RGBA")
            icon.thumbnail((112, 112), Image.Resampling.LANCZOS)
            icons.append(icon)
        except Exception as exc:
            log.warning("Cannot load icon for %s: %s", item.get("name"), exc)
            return None

    DRAFTS.mkdir(parents=True, exist_ok=True)
    image, draw = draw_background(cfg)
    accent = cfg.get("brand", {}).get("accent", "#A78BFA")
    draw.rounded_rectangle((70, 190, 420, 242), radius=24, fill=accent)
    draw.text((92, 203), f"СБОРКА  /  {build['patch_version']}", font=font(18, True), fill="#10111A")
    draw.text((70, 280), str(build["hero"]).upper(), font=font(66, True), fill="#FFFFFF")
    draw.text((72, 365), "6 ПРЕДМЕТОВ  •  ПОРЯДОК ПО СХЕМЕ", font=font(20, True), fill=accent)

    start_x, start_y = 70, 430
    cell_w, cell_h, gap_x, gap_y = 300, 205, 28, 22
    for idx, (item, icon) in enumerate(zip(items, icons)):
        row, col = divmod(idx, 3)
        x = start_x + col * (cell_w + gap_x)
        y = start_y + row * (cell_h + gap_y)
        draw.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=22, fill=(20, 22, 39, 245), outline=(167, 139, 250, 110), width=2)
        draw.ellipse((x + 18, y + 18, x + 58, y + 58), fill=accent)
        draw.text((x + 31, y + 24), str(idx + 1), font=font(20, True), fill="#11121B")
        icon_x, icon_y = x + 94, y + 24
        image.alpha_composite(icon, (icon_x, icon_y)) if image.mode == "RGBA" else image.paste(icon, (icon_x, icon_y), icon)
        item_font = font(19, True)
        item_lines = wrap_by_pixels(draw, str(item["name"]), item_font, cell_w - 28)
        text_y = y + 145
        for line in item_lines[:2]:
            draw.text((x + 14, text_y), line, font=item_font, fill="#FFFFFF")
            text_y += 25

    draw.text((70, 1125), f"Проверено: {str(build['verified_at'])[:10]}", font=font(20, True), fill="#FFFFFF")
    draw.text((70, 1165), "Сверяйся с текущим патчем перед матчем.", font=font(19), fill=(220, 220, 238, 230))
    out = DRAFTS / f"build-{re.sub(r'[^a-zA-Z0-9_-]+', '-', str(build['hero']).lower())}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.png"
    image.save(out, "PNG", optimize=True)
    return out

def telegram_send_photo(image_path: Path, caption: str) -> None:
    if not BOT_TOKEN or not CHANNEL_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID are required to publish")
    endpoint = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with image_path.open("rb") as photo:
        response = requests.post(
            endpoint,
            data={"chat_id": CHANNEL_ID, "caption": caption[:1024]},
            files={"photo": (image_path.name, photo, "image/png")},
            timeout=45,
        )
    response.raise_for_status()
    result = response.json()
    if not result.get("ok"):
        raise RuntimeError("Telegram API did not confirm sendPhoto")

def record_preview(image_path: Path, caption: str, category: str) -> None:
    preview_path = image_path.with_suffix(".txt")
    preview_path.write_text(
        f"CATEGORY: {category}\nPUBLISH_ENABLED: false\n\n{caption}\n",
        encoding="utf-8",
    )
    log.info("PREVIEW ONLY — category=%s image=%s caption=%s", category, image_path, caption.replace("\n", " | "))

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
    names = "\n".join(f"{i}. {item['name']}" for i, item in enumerate(build["items"], 1))
    return (
        f"⚔️ {build['hero']} — сборка предметов\n"
        f"Патч: {build['patch_version']}\n\n"
        f"{names}\n\n"
        f"Проверено: {build['verified_at']}\n"
        f"Источник: {build['source_url']}\n\n"
        "#MobileLegends #MLBB #Mobalistic"
    )[:1024]

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

    builds = eligible_builds()
    candidates = collect_candidates(cfg, published)
    available: set[str] = {"memes"}
    if builds:
        available.add("builds")
    for candidate in candidates:
        if candidate.category in {"guides", "news", "esports", "hot_topic"}:
            available.add(candidate.category)
    if not candidates and not builds:
        log.info("No recent trusted news/build found; only an original meme may be prepared.")
    
    posted = 0
    used_urls: set[str] = set()
    meme_key = f"meme:{datetime.now(timezone.utc).date().isoformat()}"
    while posted < MAX_POSTS_PER_RUN:
        remaining = [c for c in candidates if c.url not in used_urls]
        choices = {"memes"} if meme_key not in published and meme_key not in used_urls else set()
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
            selected = next((b for b in builds if f"build:{b['hero']}:{b['patch_version']}" not in published), None)
            if selected is None:
                builds = []
                continue
            image_path = create_build_card(cfg, selected)
            if image_path is None:
                builds = [b for b in builds if b is not selected]
                continue
            caption = build_caption(selected)
            source_key = f"build:{selected['hero']}:{selected['patch_version']}"
            builds = [b for b in builds if b is not selected]
        elif category == "memes":
            try:
                meme_title, caption = make_meme_post()
                image_path = create_editorial_card(cfg, "memes", meme_title, "ORIGINAL MOBALISTIC MEME")
                source_key = meme_key
                used_urls.add(meme_key)
            except Exception as exc:
                log.warning("Could not generate meme: %s", exc)
                used_urls.add(meme_key)
                continue
        else:
            selected_candidate = next((c for c in remaining if c.category == category), None)
            if selected_candidate is None:
                continue
            try:
                caption = make_caption(selected_candidate)
            except Exception as exc:
                log.warning("AI generation failed; skip source %s: %s", selected_candidate.url, exc)
                used_urls.add(selected_candidate.url)
                continue
            used_urls.add(selected_candidate.url)
            if not caption:
                continue
            image_path = create_editorial_card(cfg, category, selected_candidate.title, selected_candidate.publisher)
            source_key = selected_candidate.url

        if image_path is None or not caption:
            continue
        if PUBLISH_ENABLED:
            try:
                telegram_send_photo(image_path, caption)
            except Exception:
                log.exception("Telegram publication failed; state was not advanced")
                return 3
            published.add(source_key)
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

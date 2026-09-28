"""Draws one PNG prediction card per fixture for the "Share Cards to
Telegram" button (telegram_service.py's send_card_albums). Same
information as the text digest (telegram_service.py's build_digest), drawn
instead of written -- there's no browser DOM to screenshot from a FastAPI
process, so this hand-draws via Pillow rather than adding a headless-browser
dependency. Colors match frontend/src/index.css's :root exactly, so a card
looks like the same app, not a different design system."""

from __future__ import annotations

import time
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from app.schemas import FixtureOut, MatchOut
from app.services.telegram_service import pick_label, selection_label, send_telegram_media_group

ALBUM_SIZE = 10  # Telegram's sendMediaGroup max

CARD_W, CARD_H = 760, 400
PAD = 28

# frontend/src/index.css :root (light theme -- always light here regardless
# of the viewer's Telegram theme, for readability).
SURFACE = "#fcfcfb"
BORDER = "#d9d8d2"  # ~rgba(11,11,11,0.1) flattened onto SURFACE
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
SUCCESS = "#0ca30c"
ACCENT = "#c98a00"

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        # Native/local dev without the Debian font package installed --
        # falls back to Pillow's bundled bitmap font so this still runs,
        # just without a scalable typeface. Docker always has the real one.
        return ImageFont.load_default(size=size)


def build_fixture_card_image(item: FixtureOut | MatchOut) -> bytes:
    img = Image.new("RGB", (CARD_W, CARD_H), SURFACE)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, CARD_W - 1, CARD_H - 1], outline=BORDER, width=2)

    f_small = _font(20)
    f_small_bold = _font(20, bold=True)
    f_teams = _font(34, bold=True)
    f_stats = _font(22)
    f_footer = _font(16)

    x = PAD
    y = PAD
    d.text((x, y), item.league, font=f_small, fill=TEXT_MUTED)

    is_finished = getattr(item, "home_goals", None) is not None
    # Plain hyphen, not the en-dash the text digest uses -- Telegram renders
    # the text message with its own font stack, but this gets rasterized
    # with whatever font Pillow finds, and not every font covers U+2013.
    header_right = f"Final: {item.home_goals}-{item.away_goals}" if is_finished else str(item.date)
    rw = d.textlength(header_right, font=f_small_bold if is_finished else f_small)
    d.text(
        (CARD_W - PAD - rw, y),
        header_right,
        font=f_small_bold if is_finished else f_small,
        fill=TEXT_PRIMARY if is_finished else TEXT_MUTED,
    )

    y += 42
    teams = f"{item.home_team} vs {item.away_team}"
    # Shrink the team-name font if a long pair of names would overflow the
    # card width, rather than clipping or wrapping mid-name.
    tf = f_teams
    while d.textlength(teams, font=tf) > CARD_W - 2 * PAD and tf.size > 18:
        tf = _font(tf.size - 2, bold=True)
    d.text((x, y), teams, font=tf, fill=TEXT_PRIMARY)
    y += tf.size + 24

    pred = item.prediction
    if pred is None:
        d.text((x, y), "No prediction" if not is_finished else "No prediction was recorded", font=f_stats, fill=TEXT_MUTED)
    else:
        pick = pick_label(item.home_team, item.away_team, pred.match_result)
        stat_parts = []
        if pick:
            stat_parts.append(f"Pick: {pick}")
        if pred.over_under and pred.over_under.get("over_2.5") is not None:
            stat_parts.append(f"O2.5 {pred.over_under['over_2.5'] * 100:.0f}%")
        if pred.btts and pred.btts.get("yes") is not None:
            stat_parts.append(f"BTTS {pred.btts['yes'] * 100:.0f}%")
        d.text((x, y), "   ·   ".join(stat_parts), font=f_stats, fill=TEXT_SECONDARY)
        y += f_stats.size + 20

        if pred.is_value_bet and pred.value_selection:
            ev = (pred.expected_value or {}).get(pred.value_selection)
            odds = (pred.bookmaker_odds or {}).get(pred.value_selection)
            selection = selection_label(pred.value_selection, item.home_team, item.away_team)
            badge = f"VALUE: {selection}"
            if odds is not None:
                badge += f" @ {odds:.2f}"
            if ev is not None:
                badge += f"  (+{ev * 100:.1f}% EV)"
            bw = d.textlength(badge, font=f_small_bold) + 24
            d.rounded_rectangle([x, y, x + bw, y + 40], radius=8, outline=SUCCESS, width=2)
            d.text((x + 12, y + 8), badge, font=f_small_bold, fill=SUCCESS)

    footer = "Predict Model"
    fw = d.textlength(footer, font=f_footer)
    d.text((CARD_W - PAD - fw, CARD_H - PAD - f_footer.size), footer, font=f_footer, fill=TEXT_MUTED)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def send_card_albums(items: list[FixtureOut | MatchOut]) -> int:
    """Builds one card image per fixture and sends them as Telegram albums,
    ALBUM_SIZE (10) per message -- sendMediaGroup's max. Returns the number
    of album messages sent. TelegramNotConfigured propagates from
    send_telegram_media_group (checked there, not duplicated here)."""
    groups = [items[i : i + ALBUM_SIZE] for i in range(0, len(items), ALBUM_SIZE)]
    for i, group in enumerate(groups):
        media_items = [
            (f"card_{j}.png", build_fixture_card_image(item), f"{item.home_team} vs {item.away_team}")
            for j, item in enumerate(group)
        ]
        send_telegram_media_group(media_items)
        if i < len(groups) - 1:
            time.sleep(1)
    return len(groups)

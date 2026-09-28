"""Formats a fixture/prediction digest and pushes it to a Telegram chat via
the Bot API. Used by POST /api/share/telegram. requests is reused rather
than adding a second HTTP client -- it's already the only one this backend
uses (see scrape_service.py's _get())."""

from __future__ import annotations

import time

import requests

from app.config import settings
from app.schemas import FixtureOut, MatchOut

TELEGRAM_API = "https://api.telegram.org"
# Telegram's sendMessage hard limit is 4096 characters; stay well under it
# so a chunk boundary never lands mid-block.
CHUNK_CHAR_LIMIT = 3500


class TelegramNotConfigured(Exception):
    pass


# Mirrors frontend/src/components/ValueBadge.tsx's selectionLabel() so the
# digest reads the same way the UI does, not raw internal keys.
def _selection_label(key: str, home_team: str, away_team: str) -> str:
    if key == "home_win":
        return home_team
    if key == "away_win":
        return away_team
    if key == "draw":
        return "Draw"
    if key == "btts_yes":
        return "BTTS Yes"
    if key == "btts_no":
        return "BTTS No"
    side, _, line = key.partition("_")
    return f"{'Over' if side == 'over' else 'Under'} {line}"


def _pick_label(home_team: str, away_team: str, match_result: dict | None) -> str | None:
    if not match_result:
        return None
    best = max(match_result, key=match_result.get)
    label = _selection_label(best, home_team, away_team)
    return f"{label} ({match_result[best] * 100:.0f}%)"


def _fixture_block(item: FixtureOut | MatchOut) -> str:
    lines = [f"*{item.league}*", f"{item.home_team} vs {item.away_team}"]
    is_finished = getattr(item, "home_goals", None) is not None
    if is_finished:
        lines.append(f"Final: {item.home_goals}–{item.away_goals}")

    pred = item.prediction
    if pred is None:
        if not is_finished:
            lines.append("No prediction yet")
        return "\n".join(lines)

    parts = []
    pick = _pick_label(item.home_team, item.away_team, pred.match_result)
    if pick:
        parts.append(f"Pick: {pick}")
    if pred.over_under and pred.over_under.get("over_2.5") is not None:
        parts.append(f"O2.5 {pred.over_under['over_2.5'] * 100:.0f}%")
    if pred.btts and pred.btts.get("yes") is not None:
        parts.append(f"BTTS {pred.btts['yes'] * 100:.0f}%")
    if parts:
        lines.append(" · ".join(parts))

    if pred.is_value_bet and pred.value_selection:
        ev = (pred.expected_value or {}).get(pred.value_selection)
        odds = (pred.bookmaker_odds or {}).get(pred.value_selection)
        ev_str = f" (+{ev * 100:.1f}% EV)" if ev is not None else ""
        odds_str = f" @ {odds:.2f}" if odds is not None else ""
        selection = _selection_label(pred.value_selection, item.home_team, item.away_team)
        lines.append(f"\U0001F4B0 Value: {selection}{odds_str}{ev_str}")

    return "\n".join(lines)


def _chunk_blocks(blocks: list[str], limit: int = CHUNK_CHAR_LIMIT) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        block_len = len(block) + 2
        if current and current_len + block_len > limit:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(block)
        current_len += block_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def build_digest(upcoming: list[FixtureOut], finished: list[MatchOut]) -> list[str]:
    """One message chunk per return item, date-grouped, under Telegram's
    character limit. Empty list means nothing to share."""
    items: list[FixtureOut | MatchOut] = [*upcoming, *finished]
    items.sort(key=lambda f: (f.date, f.league))
    if not items:
        return []

    blocks: list[str] = []
    current_date = None
    for item in items:
        if item.date != current_date:
            current_date = item.date
            blocks.append(f"\U0001F4C5 *{current_date.strftime('%A, %d %B %Y')}*")
        blocks.append(_fixture_block(item))

    return _chunk_blocks(blocks)


def send_telegram_message(text: str) -> None:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        raise TelegramNotConfigured("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in backend/.env")
    resp = requests.post(
        f"{TELEGRAM_API}/bot{settings.telegram_bot_token}/sendMessage",
        json={"chat_id": settings.telegram_chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=15,
    )
    resp.raise_for_status()


def send_digest(chunks: list[str]) -> int:
    """Sends each chunk as its own message, 1s apart -- Telegram's per-chat
    rate limit. Returns the number of messages sent."""
    for i, chunk in enumerate(chunks):
        send_telegram_message(chunk)
        if i < len(chunks) - 1:
            time.sleep(1)
    return len(chunks)

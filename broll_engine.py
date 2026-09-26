"""
B-roll resolution: local cache first, Pexels fallback, cross-video dedup.

DELIBERATELY SEQUENTIAL. Runs BEFORE the parallel FFmpeg render phase.
Keeping this phase single-threaded is what makes the token-bucket rate
limiter below correct and avoids the SQLite write races and Pexels
rate-limit bursts found in review.

Dedup is tracked via a join table (video_id, clip_id), NOT a JSON blob
column -- a blob column read-modify-written across scenes is a lost-update
race even in a single-threaded loop with retries.

Candidate handling: Pexels returns up to 3 results per keyword. Earlier
versions only inspected videos[0], which meant a keyword was abandoned
(falling through to the next ranked keyword, or failing outright) the
moment its top result had already been used -- even though videos[1] or
videos[2] were perfectly good, unused alternatives. This version walks
every candidate for a keyword before giving up on it.
"""
from __future__ import annotations

import logging
import http.client
import sqlite3
import time
from pathlib import Path

import requests

from config import SETTINGS

log = logging.getLogger(__name__)

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
DEDUP_LOOKBACK_VIDEOS = 5  # deprioritize clips used in the last N videos channel-wide
NETWORK_RETRY_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    http.client.RemoteDisconnected,
)
MAX_NETWORK_ATTEMPTS = 3


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SETTINGS.db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_clips (
                clip_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                local_path TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS video_clip_usage (
                video_id TEXT NOT NULL,
                clip_id TEXT NOT NULL,
                used_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (video_id, clip_id),
                FOREIGN KEY (clip_id) REFERENCES stock_clips(clip_id)
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS video_builds (
                video_id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)


class RateLimiter:
    """Token-bucket limiter. Safe here because resolution is sequential --
    do not reuse across concurrent workers without a lock."""

    def __init__(self, max_per_hour: int) -> None:
        self.min_interval = 3600.0 / max_per_hour
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


_rate_limiter = RateLimiter(SETTINGS.pexels_max_req_per_hour)


def _recently_used_clip_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("""
        SELECT DISTINCT clip_id FROM video_clip_usage
        WHERE video_id IN (
            SELECT video_id FROM video_builds
            ORDER BY created_at DESC LIMIT ?
        )
    """, (DEDUP_LOOKBACK_VIDEOS,)).fetchall()
    return {r[0] for r in rows}


def _local_cache_lookup(conn: sqlite3.Connection, keyword: str, exclude_ids: set[str]) -> tuple[str, Path] | None:
    rows = conn.execute(
        "SELECT clip_id, local_path FROM stock_clips WHERE query = ?", (keyword,)
    ).fetchall()
    for clip_id, local_path in rows:
        if clip_id not in exclude_ids and Path(local_path).exists():
            return clip_id, Path(local_path)
    return None


def _pexels_fetch_candidates(keyword: str) -> list[dict]:
    """Return portrait candidates, falling back to landscape when empty."""
    headers = {"Authorization": SETTINGS.pexels_api_key}
    for orientation in ("portrait", "landscape"):
        for attempt in range(1, MAX_NETWORK_ATTEMPTS + 1):
            _rate_limiter.wait()
            try:
                resp = requests.get(
                    PEXELS_SEARCH_URL,
                    headers=headers,
                    params={
                        "query": keyword,
                        "orientation": orientation,
                        "size": "large",
                        "per_page": 5,
                    },
                    timeout=20,
                )
                resp.raise_for_status()
                videos = resp.json().get("videos", [])
                break
            except NETWORK_RETRY_EXCEPTIONS:
                if attempt == MAX_NETWORK_ATTEMPTS:
                    raise
                backoff = 2 * attempt
                log.warning(
                    "Pexels connection reset on query '%s', retrying in %ds...",
                    keyword,
                    backoff,
                )
                time.sleep(backoff)
        if videos:
            return videos
        if orientation == "portrait":
            log.info("No portrait B-roll for %r; retrying with landscape footage", keyword)
    return []


def _download_clip(
    video: dict, dest_dir: Path, output_name: str | None = None,
) -> tuple[str, Path, int, int]:
    files = sorted(
        (f for f in video["video_files"] if f.get("width")),
        key=lambda f: f["width"], reverse=True,
    )
    best = files[0]
    clip_id = str(video["id"])
    out_path = dest_dir / (output_name or f"{clip_id}.mp4")

    if not out_path.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = out_path.with_name(f"{out_path.name}.part")
        for attempt in range(1, MAX_NETWORK_ATTEMPTS + 1):
            response = None
            try:
                response = requests.get(best["link"], timeout=60, stream=True)
                response.raise_for_status()
                with temporary_path.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            output.write(chunk)
                temporary_path.replace(out_path)
                break
            except NETWORK_RETRY_EXCEPTIONS:
                temporary_path.unlink(missing_ok=True)
                if attempt == MAX_NETWORK_ATTEMPTS:
                    raise
                backoff = 2 * attempt
                log.warning(
                    "B-roll download for clip %s disconnected, retrying in %ds...",
                    clip_id,
                    backoff,
                )
                time.sleep(backoff)
            finally:
                if response is not None:
                    response.close()

    return clip_id, out_path, best["width"], best["height"]


def resolve_clip_for_scene(
    video_id: str,
    keywords: list[str],
    used_this_video: set[str],
    dest_dir: Path,
    output_name: str | None = None,
) -> Path:
    """Try each ranked keyword in order. For each keyword: local cache
    first (excluding already-used clips), then walk EVERY Pexels candidate
    for that keyword before falling through to the next keyword."""
    with _connect() as conn:
        exclude = used_this_video | _recently_used_clip_ids(conn)

        for keyword in keywords:
            hit = _local_cache_lookup(conn, keyword, exclude)
            if hit:
                clip_id, path = hit
                _register_usage(conn, video_id, clip_id)
                used_this_video.add(clip_id)
                return path

        for keyword in keywords:
            candidates = _pexels_fetch_candidates(keyword)
            for video in candidates:
                clip_id = str(video["id"])
                if clip_id in exclude:
                    # this specific candidate was already used -- try the
                    # next-ranked candidate for the SAME keyword before
                    # giving up on it entirely
                    continue

                clip_id, path, w, h = _download_clip(video, dest_dir, output_name)
                conn.execute(
                    "INSERT OR IGNORE INTO stock_clips (clip_id, query, local_path, width, height) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (clip_id, keyword, str(path), w, h),
                )
                _register_usage(conn, video_id, clip_id)
                used_this_video.add(clip_id)
                return path
            # every candidate for this keyword was already used somewhere --
            # only now move on to the next ranked keyword

    raise RuntimeError(f"No B-roll found for keywords {keywords} (all candidates exhausted)")


def _register_usage(conn: sqlite3.Connection, video_id: str, clip_id: str) -> None:
    conn.execute("INSERT OR IGNORE INTO video_builds (video_id) VALUES (?)", (video_id,))
    conn.execute(
        "INSERT OR IGNORE INTO video_clip_usage (video_id, clip_id) VALUES (?, ?)",
        (video_id, clip_id),
    )
    conn.commit()


def resolve_all_broll(
    video_id: str,
    scenes: list,
    dest_dir: Path,
    durations: dict[int, float | None] | None = None,
) -> dict[int, list[Path]]:
    """Sequential resolution phase for every scene in one video build.
    Downloads land in the CALLING project's raw_footage dir; a clip
    reused later from the local cache may physically live in a different
    project's folder (the DB row just points at wherever it was first
    downloaded) -- that's expected and fine, the cache is intentionally
    cross-project."""
    init_db()
    used_this_video: set[str] = set()
    paths: dict[int, list[Path]] = {}

    for scene in scenes:
        if len(scene.broll_keywords) < 2:
            raise RuntimeError(f"Scene {scene.scene_id} needs at least two B-roll keywords")

        scene_paths: list[Path] = []
        for index, keyword in enumerate(scene.broll_keywords[:2]):
            suffix = "a" if index == 0 else "b"
            path = resolve_clip_for_scene(
                video_id,
                [keyword],
                used_this_video,
                dest_dir,
                output_name=f"clip_{scene.scene_id:03d}_{suffix}.mp4",
            )
            used_this_video.add(path.stem)
            scene_paths.append(path)
        paths[scene.scene_id] = scene_paths
        log.info("Scene %d -> %s", scene.scene_id, ", ".join(path.name for path in scene_paths))

    return paths

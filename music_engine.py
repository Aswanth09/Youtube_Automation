"""Offline-first curated music selection with recent-track deduplication."""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from config import SETTINGS
from schemas import MusicQuery

log = logging.getLogger(__name__)

MUSIC_LIBRARY_DIR = Path(__file__).resolve().parent / "assets" / "music_library"
DB_PATH = SETTINGS.db_path
SUPPORTED_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
RECENT_TRACK_LIMIT = 5


@dataclass(frozen=True)
class MusicTrack:
    track_id: str
    path: Path
    mood: str
    tempo: str = "medium"


def _track_id(path: Path, supplied_id: str | None = None) -> str:
    if supplied_id:
        return supplied_id
    try:
        identifier = path.relative_to(MUSIC_LIBRARY_DIR).as_posix()
    except ValueError:
        identifier = path.name
    return identifier.casefold()


def _read_metadata(root: Path) -> list[MusicTrack]:
    metadata_path = root / "library.json"
    if not metadata_path.exists():
        return []

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("tracks"), list):
        records = data["tracks"]
    elif isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = [
            {"mood": mood, **record}
            for mood, mood_records in data.items()
            if isinstance(mood_records, list)
            for record in mood_records
            if isinstance(record, dict)
        ]
    else:
        raise ValueError(f"Unsupported music catalog format: {metadata_path}")

    tracks = []
    for record in records:
        if not isinstance(record, dict):
            continue
        file_value = record.get("path") or record.get("file") or record.get("filename")
        mood_value = record.get("mood")
        if not file_value or not mood_value:
            continue
        moods = mood_value if isinstance(mood_value, list) else [mood_value]
        path = Path(file_value)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            continue
        supplied_id = record.get("id") or record.get("track_id")
        for mood in moods:
            tracks.append(
                MusicTrack(
                    track_id=_track_id(path, str(supplied_id) if supplied_id else None),
                    path=path,
                    mood=str(mood),
                    tempo=str(record.get("tempo", "medium")).casefold(),
                )
            )
    return tracks


def _scan_local_tracks(root: Path) -> list[MusicTrack]:
    tracks = []
    if not root.is_dir():
        return tracks
    for mood_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(mood_dir.iterdir()):
            if path.is_file() and path.suffix.casefold() in SUPPORTED_AUDIO_SUFFIXES:
                tempo = "fast" if "fast" in path.stem.casefold() else "medium"
                tracks.append(MusicTrack(_track_id(path), path, mood_dir.name, tempo))
    return tracks


def _catalog(root: Path) -> list[MusicTrack]:
    by_path: dict[tuple[str, str], MusicTrack] = {}
    for track in (*_scan_local_tracks(root), *_read_metadata(root)):
        key = (str(track.path.resolve()).casefold(), track.mood.casefold())
        by_path[key] = track
    return list(by_path.values())


def _initialize_usage_db(connection: sqlite3.Connection) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS music_track_usage (
            selection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id TEXT NOT NULL,
            selected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)


def _recent_track_ids(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT track_id FROM music_track_usage ORDER BY selection_id DESC LIMIT ?",
        (RECENT_TRACK_LIMIT,),
    ).fetchall()
    return {str(row[0]) for row in rows}


def _copy_to_output(track: MusicTrack, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in track.track_id)
    destination = output_dir / f"music_bed_{safe_id}{track.path.suffix.lower()}"
    if track.path.resolve() != destination.resolve():
        shutil.copy2(track.path, destination)
    return destination


def ensure_music_bed(plan_music: MusicQuery, output_dir: Path) -> Path:
    """Copy a best-fit local track into output_dir and record its use."""
    root = MUSIC_LIBRARY_DIR
    catalog = _catalog(root)
    mood_tracks = [track for track in catalog if track.mood.casefold() == plan_music.mood.casefold()]
    mood_tracks.sort(key=lambda track: (track.tempo != plan_music.tempo, track.track_id.casefold()))

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH, timeout=30)) as connection:
        connection.execute("PRAGMA busy_timeout=30000")
        _initialize_usage_db(connection)
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        recently_used = _recent_track_ids(connection)
        track = next((candidate for candidate in mood_tracks if candidate.track_id not in recently_used), None)
        if track is None:
            fallback_path = root / "fallback.mp3"
            if not fallback_path.is_file():
                raise FileNotFoundError(
                    f"No unconsumed track for mood {plan_music.mood!r}, and fallback is missing: {fallback_path}"
                )
            fallback_id = "fallback"
            log.warning("Pool exhausted or track recently used; using fallback track")
            track = MusicTrack(fallback_id, fallback_path, plan_music.mood, plan_music.tempo)

        output_path = _copy_to_output(track, output_dir)
        connection.execute("INSERT INTO music_track_usage (track_id) VALUES (?)", (track.track_id,))
        connection.commit()

    log.info("Selected local music track %s for mood=%s tempo=%s", track.track_id, plan_music.mood, plan_music.tempo)
    return output_path

def prepare_project_music(project_dir: Path, mood: str) -> Path:
    """Prepare background music for the project."""
    music_dir = project_dir / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    
    # Purge any 1-second placeholder files in projects/<slug>/music/
    for f in music_dir.glob("*.mp3"):
        try:
            # Delete tiny files likely to be placeholders
            if f.stat().st_size < 50000:
                f.unlink()
        except OSError:
            pass

    assets_music = Path(__file__).resolve().parent / "assets" / "music"
    lib_music = project_dir.parent / "_library"
    
    selected_track = None
    for search_dir in [assets_music, lib_music]:
        if not search_dir.exists():
            continue
        for track in search_dir.rglob("*.mp3"):
            if mood.casefold() in track.name.casefold():
                selected_track = track
                break
        if selected_track:
            break
            
    if not selected_track:
        # Fallback to any available .mp3
        for search_dir in [assets_music, lib_music]:
            if not search_dir.exists():
                continue
            any_track = next(search_dir.rglob("*.mp3"), None)
            if any_track:
                selected_track = any_track
                break

    bg_music_path = music_dir / "background.mp3"
    if selected_track and selected_track.exists():
        shutil.copy2(selected_track, bg_music_path)
        
    return bg_music_path

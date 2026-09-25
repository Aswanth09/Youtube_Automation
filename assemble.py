"""
Concatenate rendered scenes and lay a ducked music bed under the result.

The music bed is no longer a manual CLI argument -- Gemini Stage 2 already
picked a `suggested_music_mood`, so this module just maps that mood to a
file under assets/music/ and loads it automatically.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import requests
import imageio_ffmpeg

log = logging.getLogger(__name__)

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
MUSIC_DIR = Path(__file__).resolve().parent / "assets" / "music"
FMA_API_URL = "https://freemusicarchive.org/api/get/tracks.json"
MUSIC_SEARCH_TERMS = {
    "corporate_tension": "ambient corporate minimal tension instrumental",
    "dark_suspense": "low drone suspense instrumental",
    "slow_investigation": "steady rhythmic investigative instrumental",
}

DUCK_FILTER = (
    "[1:a]volume=0.18[bed];"
    "[0:a][bed]sidechaincompress=threshold=0.125:ratio=4:attack=20:release=300:makeup=1[bedducked];"
    "[0:a][bedducked]amix=inputs=2:duration=first:dropout_transition=2[aout]"
)

MOOD_TRACK_MAP = {
    "corporate_tension": "corporate_tension.mp3",
    "dark_suspense": "dark_suspense.mp3",
    "slow_investigation": "slow_investigation.mp3",
}


def resolve_music_track(mood: str) -> Path:
    filename = MOOD_TRACK_MAP.get(mood)
    if not filename:
        raise ValueError(f"Unknown music mood: {mood!r} (expected one of {list(MOOD_TRACK_MAP)})")

    path = MUSIC_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    try:
        _download_music_track(mood, path)
    except Exception as error:
        log.warning("Music download failed for %s: %s", mood, error)

    if not path.exists() or path.stat().st_size == 0:
        _create_silent_music_bed(path)
    return path


def _download_music_track(mood: str, destination: Path) -> None:
    """Find and stream a mood-matched FMA track into the local cache."""
    api_key = os.getenv("FMA_API_KEY")
    if not api_key:
        raise RuntimeError("FMA_API_KEY is not configured")

    response = requests.get(
        FMA_API_URL,
        params={
            "api_key": api_key,
            "q": MUSIC_SEARCH_TERMS[mood],
            "limit": 10,
            "sort": "track_date_published",
            "sort_dir": "desc",
        },
        timeout=(5, 15),
    )
    response.raise_for_status()
    tracks = response.json().get("dataset", response.json().get("tracks", []))
    if not isinstance(tracks, list):
        raise RuntimeError("FMA returned an unexpected track list")

    download_url = next(
        (
            track.get("track_file") or track.get("download_url")
            for track in tracks
            if isinstance(track, dict) and (track.get("track_file") or track.get("download_url"))
        ),
        None,
    )
    if not download_url:
        raise RuntimeError(f"FMA returned no downloadable track for {mood}")

    temporary_path = destination.with_suffix(".download")
    try:
        with requests.get(download_url, stream=True, timeout=(5, 15)) as audio_response:
            audio_response.raise_for_status()
            total_bytes = int(audio_response.headers.get("content-length", 0))
            downloaded_bytes = 0
            with temporary_path.open("wb") as output:
                for chunk in audio_response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    output.write(chunk)
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes // (1024 * 1024) != (downloaded_bytes - len(chunk)) // (1024 * 1024):
                        if total_bytes:
                            log.info(
                                "Downloading %s: %.1f%%",
                                destination.name,
                                downloaded_bytes * 100 / total_bytes,
                            )
                        else:
                            log.info("Downloading %s: %.1f MB", destination.name, downloaded_bytes / 1048576)
        if temporary_path.stat().st_size == 0:
            raise RuntimeError("downloaded track was empty")
        temporary_path.replace(destination)
        log.info("Cached music track: %s", destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _create_silent_music_bed(destination: Path) -> None:
    """Create a valid short MP3 that add_music_bed can loop indefinitely."""
    subprocess.run(
        [
            FFMPEG_BIN, "-y",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-t", "1", "-c:a", "libmp3lame", "-b:a", "128k",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    log.warning("Using silent fallback music bed: %s", destination)


def concat_scenes(scene_files: list[Path], out_path: Path) -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in scene_files:
            f.write(f"file '{p.resolve()}'\n")
        list_path = f.name

    subprocess.run(
        [FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", str(out_path)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    return out_path


def add_music_bed(concat_path: Path, music_bed_path: Path, final_out_path: Path) -> Path:
    subprocess.run(
        [
            FFMPEG_BIN, "-y",
            "-i", str(concat_path),
            "-stream_loop", "-1", "-i", str(music_bed_path),
            "-filter_complex", DUCK_FILTER,
            "-map", "0:v", "-map", "[aout]",
            "-shortest",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            str(final_out_path),
        ],
        check=True, capture_output=True, text=True, timeout=180,
    )
    return final_out_path


def assemble_final_video(slug: str, scene_files: list[Path], mood: str, out_dir: Path) -> Path:
    music_bed_path = resolve_music_track(mood)

    concat_only = out_dir / f"{slug}_concat_novoice_bed.mp4"
    final_out = out_dir / f"{slug}_final.mp4"

    concat_scenes(scene_files, concat_only)
    add_music_bed(concat_only, music_bed_path, final_out)

    concat_only.unlink(missing_ok=True)
    return final_out

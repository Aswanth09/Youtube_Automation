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
import wave
from pathlib import Path

import requests
import imageio_ffmpeg

from config import AUDIO_CHANNELS, INTER_TURN_GAP_MS, SAMPLE_RATE

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
    "[0:a]asplit=2[voice][key];"
    "[1:a]volume=0.45[bed];"
    "[bed][key]sidechaincompress=threshold=0.03:ratio=8:attack=15:release=450:makeup=1[ducked];"
    "[voice][ducked]amix=inputs=2:duration=first:normalize=0[mix];"
    "[mix]loudnorm=I=-14:TP=-1.5:LRA=11[aout]"
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
            "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
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
            "-ar", str(SAMPLE_RATE), "-ac", str(AUDIO_CHANNELS),
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


def assemble_final_video_with_music(
    slug: str,
    scene_files: list[Path],
    music_bed_path: Path,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    concat_only = out_dir / f"{slug}_concat_novoice_bed.mp4"
    final_out = out_dir / f"{slug}_final.mp4"
    concat_scenes(scene_files, concat_only)
    add_music_bed(concat_only, music_bed_path, final_out)
    concat_only.unlink(missing_ok=True)
    return final_out


def write_concat_list(video_files: list[Path], concat_list: Path) -> Path:
    concat_list.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for video_path in video_files:
        escaped_path = video_path.resolve().as_posix().replace("'", "'\\''")
        entries.append(f"file '{escaped_path}'")
    concat_list.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return concat_list


def concatenate_beat_videos(video_files: list[Path], output_path: Path) -> Path:
    concat_list = write_concat_list(video_files, output_path.parent / "concat_list.txt")
    subprocess.run(
        [FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c:v", "copy", "-an", str(output_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return output_path


def build_master_narration_wav(audio_files: list[Path], output_path: Path) -> Path:
    """Join sample-accurate beat WAVs with configured gaps between turns."""
    if not audio_files:
        raise ValueError("At least one beat narration WAV is required")
    gap_frames = SAMPLE_RATE * INTER_TURN_GAP_MS // 1000
    expected_format = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as master:
        for index, audio_path in enumerate(audio_files):
            with wave.open(str(audio_path), "rb") as beat_audio:
                audio_format = (
                    beat_audio.getnchannels(),
                    beat_audio.getsampwidth(),
                    beat_audio.getframerate(),
                    beat_audio.getcomptype(),
                )
                if expected_format is None:
                    expected_format = audio_format
                    if audio_format[:3] != (AUDIO_CHANNELS, 2, SAMPLE_RATE):
                        raise ValueError(f"Beat WAV must be 48 kHz 16-bit stereo: {audio_path}")
                    master.setnchannels(audio_format[0])
                    master.setsampwidth(audio_format[1])
                    master.setframerate(audio_format[2])
                elif audio_format != expected_format:
                    raise ValueError(f"Beat WAV format differs from other turns: {audio_path}")

                master.writeframes(beat_audio.readframes(beat_audio.getnframes()))
            if index < len(audio_files) - 1:
                master.writeframes(b"\x00" * gap_frames * AUDIO_CHANNELS * 2)
    return output_path


def mix_narration_and_music(
    silent_video_path: Path,
    narration_path: Path,
    music_path: Path,
    final_path: Path,
) -> Path:
    subprocess.run(
        [
            FFMPEG_BIN, "-y",
            "-i", str(silent_video_path),
            "-i", str(narration_path),
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex", DUCK_FILTER.replace("[0:a]", "[1:a]").replace("[1:a]volume", "[2:a]volume"),
            "-map", "0:v:0", "-map", "[aout]", "-shortest",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-ar", str(SAMPLE_RATE), "-ac", str(AUDIO_CHANNELS), str(final_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    return final_path


def assemble_dual_host_video(
    slug: str,
    beat_video_files: list[Path],
    narration_files: list[Path],
    music_path: Path,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    silent_video = out_dir / f"{slug}_video_only.mp4"
    master_voice = out_dir / f"{slug}_narration_master.wav"
    final_path = out_dir / f"{slug}_final.mp4"
    concatenate_beat_videos(beat_video_files, silent_video)
    build_master_narration_wav(narration_files, master_voice)
    mix_narration_and_music(silent_video, master_voice, music_path, final_path)
    silent_video.unlink(missing_ok=True)
    master_voice.unlink(missing_ok=True)
    return final_path

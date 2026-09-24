"""
Voiceover generation + timing reconciliation.

Always trust ffprobe's measured duration over any word-count estimate --
this is the step that prevents audio/caption drift from accumulating
across 16-20 concatenated scenes.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import edge_tts

from config import SETTINGS


def quantize(seconds: float) -> float:
    """Snap a duration to the nearest whole frame at SETTINGS.fps, so the
    concat demuxer never accumulates fractional-frame rounding error."""
    frames = round(seconds * SETTINGS.fps)
    return frames / SETTINGS.fps


from mutagen.mp3 import MP3

def get_audio_duration(file_path: str) -> float:
    audio = MP3(str(file_path))
    return float(audio.info.length)


async def _synthesize(text: str, out_path: Path, voice: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def generate_scene_audio(scene_id: int, narration: str, audio_dir: Path) -> tuple[Path, float]:
    """Synthesize one scene's narration into the given project's audio
    directory and return (path, quantized_duration)."""
    out_path = audio_dir / f"audio_{scene_id:03d}.mp3"
    asyncio.run(_synthesize(narration, out_path, SETTINGS.tts_voice))

    raw_duration = get_audio_duration(out_path)
    duration = quantize(raw_duration)
    return out_path, duration

"""
Voiceover generation + timing reconciliation.

Always trust ffprobe's measured duration over any word-count estimate --
this is the step that prevents audio/caption drift from accumulating
across 16-20 concatenated scenes.
"""
from __future__ import annotations

import asyncio
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


def _ass_timestamp(seconds: float) -> str:
    total_centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(total_centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    centiseconds, seconds_value = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds_value:02d}.{centiseconds:02d}"


def _ass_text(text: str) -> str:
    return text.upper().replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def _write_ass(subtitle_path: Path, boundaries: list[tuple[float, float, str]]) -> None:
    subtitle_path.write_text(
        """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,30,&H0000FFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,45,45,45,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        + "\n".join(
            f"Dialogue: 0,{_ass_timestamp(start)},{_ass_timestamp(end)},Default,,0,0,45,,{_ass_text(caption)}"
            for start, end, caption in boundaries
            if caption.strip() and end > start
        )
        + "\n",
        encoding="utf-8",
    )


async def _synthesize(text: str, out_path: Path, voice: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    word_boundaries: list[tuple[float, float, str]] = []
    sentence_boundaries: list[tuple[float, float, str]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as audio_file:
        async for message in communicate.stream():
            if message["type"] == "audio":
                audio_file.write(message["data"])
                continue

            if message["type"] not in {"WordBoundary", "SentenceBoundary"}:
                continue
            start = float(message["offset"]) / 10_000_000
            event_duration = float(message.get("duration", 0)) / 10_000_000
            end = max(start + event_duration, start + 0.05)
            boundary = (start, end, str(message.get("text", "")))
            if message["type"] == "WordBoundary":
                word_boundaries.append(boundary)
            else:
                sentence_boundaries.append(boundary)

    subtitle_path = out_path.with_name(out_path.name.replace("audio_", "subs_", 1).rsplit(".", 1)[0] + ".ass")
    _write_ass(subtitle_path, word_boundaries or sentence_boundaries)


def generate_scene_audio(scene_id: int, narration: str, audio_dir: Path) -> tuple[Path, float]:
    """Synthesize one scene's narration into the given project's audio
    directory and return (path, quantized_duration)."""
    out_path = audio_dir / f"audio_{scene_id:03d}.mp3"
    asyncio.run(_synthesize(narration, out_path, SETTINGS.tts_voice))

    raw_duration = get_audio_duration(out_path)
    duration = quantize(raw_duration)
    return out_path, duration

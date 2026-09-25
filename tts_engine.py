"""Voiceover generation, timing reconciliation, and word-level ASS captions."""
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
    seconds_value, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds_value:02d}.{centiseconds:02d}"


def _ass_text(text: str) -> str:
    return text.upper().replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def _write_ass(subtitle_path: Path, boundaries: list[tuple[float, float, str]]) -> None:
    dialogue_lines = []
    index = 0
    while index < len(boundaries):
        remaining = len(boundaries) - index
        chunk_size = 2 if remaining == 4 else min(3, remaining)
        chunk = boundaries[index:index + chunk_size]
        if not chunk:
            break
        start = chunk[0][0]
        end = chunk[-1][1]
        text = " ".join(word for _, _, word in chunk).upper()
        dialogue_lines.append(
            f"Dialogue: 0,{_ass_timestamp(start)},{_ass_timestamp(end)},Default,,0,0,0,,{_ass_text(text)}"
        )
        index += chunk_size

    subtitle_path.write_text(
        """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,56,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,8,40,40,800,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        + "\n".join(dialogue_lines)
        + "\n",
        encoding="utf-8",
    )


async def _synthesize(text: str, out_path: Path, voice: str) -> Path:
    communicate = edge_tts.Communicate(text, voice)
    submaker = edge_tts.SubMaker()
    word_boundaries: list[tuple[float, float, str]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as audio_file:
        async for message in communicate.stream():
            if message["type"] == "audio":
                audio_file.write(message["data"])
                continue

            if message["type"] != "WordBoundary":
                continue
            start = float(message["offset"]) / 10_000_000
            event_duration = float(message.get("duration", 0)) / 10_000_000
            end = max(start + event_duration, start + 0.05)
            word = str(message.get("text", "")).strip()
            if not word:
                continue
            submaker.create_sub((message["offset"], message.get("duration", 0)), word)
            word_boundaries.append((start, end, word))

    submaker.generate_subs()
    subtitle_path = out_path.with_name(out_path.name.replace("audio_", "subs_", 1).rsplit(".", 1)[0] + ".ass")
    _write_ass(subtitle_path, word_boundaries)
    return subtitle_path


def generate_scene_audio(scene_id: int, narration: str, audio_dir: Path) -> tuple[Path, Path, float]:
    """Synthesize one scene's narration into the given project's audio
    directory and return (path, quantized_duration)."""
    out_path = audio_dir / f"audio_{scene_id:03d}.mp3"
    ass_path = asyncio.run(_synthesize(narration, out_path, SETTINGS.tts_voice))

    raw_duration = get_audio_duration(out_path)
    duration = quantize(raw_duration)
    return out_path, ass_path, duration

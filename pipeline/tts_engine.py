"""Voiceover generation, timing reconciliation, and word-level ASS captions."""
from __future__ import annotations

import asyncio
import subprocess
import wave
from pathlib import Path

import edge_tts
import imageio_ffmpeg

from config import AUDIO_CHANNELS, DEFAULT_VOICE, SAMPLE_RATE, SETTINGS, VOICE_MAP

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
TICKS_PER_SECOND = 10_000_000


def _ass_timestamp(sample_offset: int) -> str:
    total_centiseconds = max(0, (sample_offset * 100 + SAMPLE_RATE // 2) // SAMPLE_RATE)
    hours, remainder = divmod(total_centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds_value, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds_value:02d}.{centiseconds:02d}"


def _ass_text(text: str) -> str:
    return text.upper().replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def _speaker_ass_style(speaker: str | None) -> str:
    """Map optional speaker labels to dedicated ASS styles."""
    return {"maya": "Maya", "jax": "Jax"}.get((speaker or "").casefold(), "Default")


def _active_word_color(speaker: str | None) -> str:
    if (speaker or "").casefold() == "maya":
        return r"\c&H00D900FF&"
    if (speaker or "").casefold() == "jax":
        return r"\c&H00FFFF00&"
    return r"\c&H0014F0FF&"


def _karaoke_chunk_text(chunk: list, active_index: int, speaker: str | None) -> str:
    active_color = _active_word_color(speaker)
    parts = []
    for index, boundary in enumerate(chunk):
        color = active_color if index == active_index else r"\c&H00FFFFFF&"
        parts.append("{" + color + "}" + _ass_text(boundary[2]))
    parts.append(r"{\c&H00FFFFFF&}")
    return " ".join(parts)


def _speaker_for_voice(voice: str) -> str | None:
    normalized_voice = voice.casefold()
    for speaker, mapped_voice in VOICE_MAP.items():
        if normalized_voice in {speaker.casefold(), mapped_voice.casefold()}:
            return speaker
    return None


def _write_ass(
    subtitle_path: Path,
    boundaries: list[tuple[int, int, str] | tuple[int, int, str, str]],
) -> None:
    dialogue_lines = []
    index = 0
    while index < len(boundaries):
        first = boundaries[index]
        speaker = first[3] if len(first) > 3 else None
        speaker_end = index + 1
        while speaker_end < len(boundaries):
            boundary = boundaries[speaker_end]
            next_speaker = boundary[3] if len(boundary) > 3 else None
            if next_speaker != speaker:
                break
            speaker_end += 1

        remaining = speaker_end - index
        chunk_size = 2 if remaining == 4 else min(3, remaining)
        chunk = boundaries[index:index + chunk_size]
        if not chunk:
            break
        style = _speaker_ass_style(speaker)
        for active_index, boundary in enumerate(chunk):
            start, end = boundary[0], boundary[1]
            if end <= start:
                continue
            karaoke_text = _karaoke_chunk_text(chunk, active_index, speaker)
            dialogue_lines.append(
                f"Dialogue: 0,{_ass_timestamp(start)},{_ass_timestamp(end)}"
                f",{style},,0,0,0,,{karaoke_text}"
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
Style: Default,Arial Black,74,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,40,40,960,1
Style: Maya,Arial Black,74,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,40,40,960,1
Style: Jax,Arial Black,74,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,40,40,960,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        + "\n".join(dialogue_lines)
        + "\n",
        encoding="utf-8",
    )


async def _synthesize(text: str, out_path: Path, voice: str) -> list[tuple[int, int, str]]:
    voice = VOICE_MAP.get(voice.casefold(), voice)
    communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    word_boundaries: list[tuple[int, int, str]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as audio_file:
        async for message in communicate.stream():
            if message["type"] == "audio":
                audio_file.write(message["data"])
                continue

            if message["type"] != "WordBoundary":
                continue
            word = str(message.get("text", "")).strip()
            if not word:
                continue
            start_ticks = int(message["offset"])
            end_ticks = int(message["offset"]) + int(message.get("duration", 0))
            start_sample = (start_ticks * SAMPLE_RATE + TICKS_PER_SECOND // 2) // TICKS_PER_SECOND
            end_sample = (end_ticks * SAMPLE_RATE + TICKS_PER_SECOND // 2) // TICKS_PER_SECOND
            word_boundaries.append((start_sample, end_sample, word))

    return word_boundaries


def _decode_pcm_wav(source_path: Path, wav_path: Path) -> int:
    subprocess.run(
        [
            FFMPEG_BIN, "-y", "-i", str(source_path),
            "-ar", str(SAMPLE_RATE), "-ac", str(AUDIO_CHANNELS),
            "-c:a", "pcm_s16le", str(wav_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with wave.open(str(wav_path), "rb") as pcm_file:
        if (
            pcm_file.getframerate() != SAMPLE_RATE
            or pcm_file.getnchannels() != AUDIO_CHANNELS
            or pcm_file.getsampwidth() != 2
        ):
            raise ValueError(f"Decoded TTS WAV has unexpected PCM format: {wav_path}")
        return pcm_file.getnframes()


def _fallback_boundaries(text: str, total_samples: int) -> list[tuple[int, int, str]]:
    words = text.split()
    if not words:
        raise ValueError("Cannot create subtitles for empty narration")
    if total_samples < len(words):
        raise ValueError("Audio is too short to allocate a sample to every narration word")
    return [
        (index * total_samples // len(words), (index + 1) * total_samples // len(words), word)
        for index, word in enumerate(words)
    ]


def generate_scene_audio(
    scene_id: int,
    narration: str,
    audio_dir: Path,
    speaker: str | None = None,
    voice: str | None = None,
) -> tuple[Path, Path, float]:
    """Synthesize one scene's narration into the given project's audio
    directory and return (PCM WAV path, ASS path, exact sample duration)."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_path = audio_dir / f"audio_{scene_id:03d}_raw.mp3"
    out_path = audio_dir / f"audio_{scene_id:03d}.wav"
    ass_path = audio_dir / f"subs_{scene_id:03d}.ass"
    if speaker:
        selected_voice = VOICE_MAP.get(speaker.casefold(), DEFAULT_VOICE)
    else:
        selected_voice = voice or DEFAULT_VOICE
    try:
        word_boundaries = asyncio.run(_synthesize(narration, raw_path, selected_voice))
        total_samples = _decode_pcm_wav(raw_path, out_path)
    finally:
        raw_path.unlink(missing_ok=True)

    if not word_boundaries:
        word_boundaries = _fallback_boundaries(narration, total_samples)
    else:
        clamped_boundaries = []
        for start, end, word in word_boundaries:
            start = min(max(0, start), total_samples - 1)
            end = min(max(start + 1, end), total_samples)
            clamped_boundaries.append((start, end, word))
        word_boundaries = clamped_boundaries

    speaker_style = speaker or _speaker_for_voice(selected_voice)
    if speaker_style:
        word_boundaries = [
            (start, end, word, speaker_style)
            for start, end, word in word_boundaries
        ]

    _write_ass(ass_path, word_boundaries)
    duration = total_samples / SAMPLE_RATE
    return out_path, ass_path, duration

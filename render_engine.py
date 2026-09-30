"""Per-scene FFmpeg rendering."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from config import (
    AUDIO_CHANNELS,
    AVATAR_ALICE,
    AVATAR_ALICE_PNG,
    AVATAR_BOB,
    AVATAR_BOB_PNG,
    AVATAR_POS_ALICE,
    AVATAR_POS_BOB,
    AVATAR_SIZE,
    SAMPLE_RATE,
    SETTINGS,
)
import imageio_ffmpeg

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

log = logging.getLogger(__name__)

RENDER_TIMEOUT_SEC = 180
VIDEO_WIDTH, VIDEO_HEIGHT = 1080, 1920
PREVIEW_WIDTH, PREVIEW_HEIGHT = VIDEO_WIDTH, VIDEO_HEIGHT
PREVIEW_SCENE_LIMIT = 2


def _subtitle_filter(subtitles_path: Path) -> str:
    if not subtitles_path.exists():
        raise FileNotFoundError(f"Missing ASS subtitles: {subtitles_path}")

    escaped_sub = str(subtitles_path.resolve()).replace("\\", "/").replace(":", r"\:")
    return f"subtitles='{escaped_sub}':fontsdir='C\\:/Windows/Fonts'"


def _vertical_motion_filter(
    total_duration: float,
    segment_duration: float,
    width: int,
    height: int,
    time_offset: float = 0.0,
) -> str:
    progress = f"(t/{total_duration})" if time_offset == 0 else f"((t+{time_offset})/{total_duration})"
    return (
        f"fps=30,scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"scale=eval=frame:w='{width}*(1+0.04*{progress})':"
        f"h='{height}*(1+0.04*{progress})',"
        f"crop={width}:{height},settb=AVTB,"
        f"trim=duration={segment_duration:.6f},setpts=PTS-STARTPTS,settb=AVTB,setsar=1"
    )


def build_beat_filtergraph(
    beat,
    duration: float,
    subtitles_path: Path,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
) -> str:
    """Build the script-file graph for two-cut video, avatar, and captions."""
    if not subtitles_path.is_file():
        raise FileNotFoundError(f"Missing ASS subtitles for beat {beat.beat_id}: {subtitles_path}")

    first_duration = duration / 2
    second_duration = duration - first_duration
    avatar_x, avatar_y = AVATAR_POS_ALICE if beat.speaker == "alice" else AVATAR_POS_BOB
    escaped_sub = str(subtitles_path.resolve()).replace("\\", "/").replace(":", r"\:")
    first_chain = _vertical_motion_filter(duration, first_duration, width, height)
    second_chain = _vertical_motion_filter(duration, second_duration, width, height, first_duration)
    pulse = f"{AVATAR_SIZE}+12*(0.5+0.5*sin(2*PI*t))"

    return (
        f"[0:v]{first_chain}[v0];\n"
        f"[1:v]{second_chain}[v1];\n"
        f"[v0][v1]concat=n=2:v=1:a=0,setsar=1,settb=AVTB[background];\n"
        f"[2:v]format=rgba,scale=w='{pulse}':h='{pulse}':eval=frame,setsar=1[avatar];\n"
        f"[background][avatar]overlay=x={avatar_x}:y={avatar_y}:eval=frame:shortest=1[with_avatar];\n"
        f"[with_avatar]subtitles='{escaped_sub}':fontsdir='C\\:/Windows/Fonts'[vout]"
    )


def render_beat(
    beat,
    broll_paths: list[Path],
    avatar_path: Path,
    subtitles_path: Path,
    duration: float,
    out_dir: Path,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
) -> Path:
    if (width, height) != (VIDEO_WIDTH, VIDEO_HEIGHT):
        raise ValueError(f"Beat render dimensions must be {VIDEO_WIDTH}x{VIDEO_HEIGHT}")
    if len(broll_paths) != 2:
        raise ValueError(f"Beat {beat.beat_id} requires exactly two B-roll clips")
    for asset_path in (*broll_paths, avatar_path, subtitles_path):
        if not asset_path.is_file():
            raise FileNotFoundError(f"Missing asset for beat {beat.beat_id}: {asset_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"beat_{beat.beat_id:03d}.mp4"
    filtergraph = build_beat_filtergraph(beat, duration, subtitles_path, width, height)
    script_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ffgraph", encoding="utf-8", delete=False) as graph_file:
            graph_file.write(filtergraph)
            script_path = Path(graph_file.name)

        cmd = [FFMPEG_BIN, "-y"]
        for clip_path in broll_paths:
            cmd.extend(["-stream_loop", "-1", "-i", str(clip_path)])
        if avatar_path.suffix.casefold() == ".gif":
            cmd.extend(["-ignore_loop", "0", "-i", str(avatar_path)])
        else:
            cmd.extend(["-loop", "1", "-framerate", "30", "-i", str(avatar_path)])
        cmd.extend([
            "-filter_complex_script", str(script_path),
            "-map", "[vout]", "-an", "-t", str(duration),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-r", "30", str(out_path),
        ])
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=RENDER_TIMEOUT_SEC)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Beat {beat.beat_id} render failed: {error.stderr[-2000:]}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Beat {beat.beat_id} render timed out after {RENDER_TIMEOUT_SEC}s") from error
    finally:
        if script_path is not None:
            script_path.unlink(missing_ok=True)
    return out_path


def render_all_beats(
    beats: list,
    broll_paths: dict[int, list[Path]],
    audio_data: dict,
    subtitles_paths: dict[int, Path],
    out_dir: Path,
    preview: bool = False,
) -> list[Path]:
    target_beats = beats[:PREVIEW_SCENE_LIMIT] if preview else beats
    max_workers = min(SETTINGS.max_render_workers, len(target_beats)) or 1
    results: dict[int, Path] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for beat in target_beats:
            _, duration = audio_data[beat.beat_id]
            if beat.speaker == "alice":
                avatar_path = AVATAR_ALICE if AVATAR_ALICE.is_file() else AVATAR_ALICE_PNG
            else:
                avatar_path = AVATAR_BOB if AVATAR_BOB.is_file() else AVATAR_BOB_PNG
            future = pool.submit(
                render_beat,
                beat,
                broll_paths[beat.beat_id],
                avatar_path,
                subtitles_paths[beat.beat_id],
                duration,
                out_dir,
            )
            futures[future] = beat.beat_id
        for future in as_completed(futures):
            beat_id = futures[future]
            results[beat_id] = future.result()
            log.info("Beat %d rendered OK", beat_id)
    return [results[beat.beat_id] for beat in target_beats]


def render_scene(
    scene, broll_path: Path | list[Path], audio_path: Path, subtitles_path: Optional[Path],
    duration: float, out_dir: Path, width: int = 1080, height: int = 1920,
    preview: bool = False,
) -> Path:
    if (width, height) != (VIDEO_WIDTH, VIDEO_HEIGHT):
        raise ValueError(f"Render dimensions must be {VIDEO_WIDTH}x{VIDEO_HEIGHT}")

    clip_paths = [broll_path] if isinstance(broll_path, Path) else list(broll_path)
    if len(clip_paths) < 2:
        raise ValueError(f"Scene {scene.scene_id} requires two B-roll clips")
    clip_paths = clip_paths[:2]
    for clip_path in clip_paths:
        if not clip_path.exists():
            raise FileNotFoundError(f"Missing B-roll for scene {scene.scene_id}: {clip_path}")

    first_duration = duration / 2
    second_duration = duration - first_duration
    filter_complex = (
        f"[0:v]{_vertical_motion_filter(duration, first_duration, width, height)}[v0];"
        f"[1:v]{_vertical_motion_filter(duration, second_duration, width, height, first_duration)}[v1];"
        f"[v0][v1]concat=n=2:v=1:a=0,setsar=1,settb=AVTB[vbase];"
        f"[vbase]{_subtitle_filter(subtitles_path)}[vout]"
    )

    out_path = out_dir / f"scene_{scene.scene_id:03d}.mp4"

    cmd = [
        FFMPEG_BIN, "-y",
    ]
    for clip_path in clip_paths:
        cmd.extend(["-stream_loop", "-1", "-i", str(clip_path)])
    cmd.extend([
        "-i", str(audio_path),
        "-filter_complex", filter_complex,
        "-t", str(duration),
        "-map", "[vout]",
        "-map", "2:a:0",
        "-shortest",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(AUDIO_CHANNELS),
        "-pix_fmt", "yuv420p",
        str(out_path),
    ])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=RENDER_TIMEOUT_SEC)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Scene {scene.scene_id} render failed: {e.stderr[-2000:]}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Scene {scene.scene_id} render timed out after {RENDER_TIMEOUT_SEC}s") from e

    return out_path


def render_all_scenes(
    scenes: list,
    broll_paths: dict[int, list[Path]],
    audio_data: dict,
    out_dir: Path,
    preview: bool = False,
    subtitles_paths: dict[int, Path] | None = None,
) -> list[Path]:
    target_scenes = scenes[:PREVIEW_SCENE_LIMIT] if preview else scenes
    width, height = VIDEO_WIDTH, VIDEO_HEIGHT
    max_workers = min(SETTINGS.max_render_workers, len(target_scenes)) or 1

    results: dict[int, Path] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for scene in target_scenes:
            audio_path, duration = audio_data[scene.scene_id]
            subtitles_path = (subtitles_paths or {}).get(scene.scene_id)
            fut = pool.submit(
                render_scene,
                scene,
                broll_paths[scene.scene_id],
                audio_path,
                subtitles_path,
                duration,
                out_dir,
                width,
                height,
                preview,
            )
            futures[fut] = scene.scene_id

        for fut in as_completed(futures):
            scene_id = futures[fut]
            try:
                results[scene_id] = fut.result()
                log.info("Scene %d rendered OK", scene_id)
            except RuntimeError as e:
                raise RuntimeError(f"Render pipeline aborted: {e}") from e

    ordered_ids = [s.scene_id for s in target_scenes]
    return [results[sid] for sid in ordered_ids]

"""Per-scene FFmpeg rendering."""
from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from config import SETTINGS
import imageio_ffmpeg

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

log = logging.getLogger(__name__)

RENDER_TIMEOUT_SEC = 180
PREVIEW_WIDTH, PREVIEW_HEIGHT = 1080, 1920
PREVIEW_SCENE_LIMIT = 2


def _subtitle_filter(subtitles_path: Optional[Path]) -> str | None:
    if subtitles_path is None or not subtitles_path.exists():
        return None

    escaped_ass = str(subtitles_path.resolve()).replace("\\", "/").replace(":", r"\:")
    return f"subtitles=filename='{escaped_ass}':fontsdir='C\\:/Windows/Fonts'"


def render_scene(
    scene, broll_path: Path, audio_path: Path, subtitles_path: Optional[Path],
    duration: float, out_dir: Path, width: int = 1080, height: int = 1920,
    preview: bool = False,
) -> Path:
    if not broll_path.exists():
        raise FileNotFoundError(f"Missing B-roll for scene {scene.scene_id}: {broll_path}")

    filter_chain = (
        f"fps=30,scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"scale=eval=frame:w='{width}*(1+0.05*(t/{duration}))':"
        f"h='{height}*(1+0.05*(t/{duration}))',"
        f"crop={width}:{height},settb=AVTB"
    )
    subtitle_filter = _subtitle_filter(subtitles_path)
    if subtitle_filter:
        filter_chain = f"{filter_chain},{subtitle_filter}"

    out_path = out_dir / f"scene_{scene.scene_id:03d}.mp4"

    cmd = [
        FFMPEG_BIN, "-y",
        "-stream_loop", "-1",
        "-i", str(broll_path),
        "-i", str(audio_path),
        "-vf", filter_chain,
        "-t", str(duration),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=RENDER_TIMEOUT_SEC)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Scene {scene.scene_id} render failed: {e.stderr[-2000:]}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Scene {scene.scene_id} render timed out after {RENDER_TIMEOUT_SEC}s") from e

    return out_path


def render_all_scenes(
    scenes: list,
    broll_paths: dict[int, Path],
    audio_data: dict,
    out_dir: Path,
    preview: bool = False,
    subtitles_paths: dict[int, Path] | None = None,
) -> list[Path]:
    target_scenes = scenes[:PREVIEW_SCENE_LIMIT] if preview else scenes
    width, height = (PREVIEW_WIDTH, PREVIEW_HEIGHT) if preview else (SETTINGS.width, SETTINGS.height)
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

"""Per-scene FFmpeg rendering."""
from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from config import SETTINGS
import imageio_ffmpeg

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

log = logging.getLogger(__name__)

RENDER_TIMEOUT_SEC = 180
PREVIEW_WIDTH, PREVIEW_HEIGHT = 1280, 720
PREVIEW_SCENE_LIMIT = 2


def _subtitle_filter(audio_path: Path) -> str:
    subtitle_path = audio_path.with_name(
        audio_path.name.replace("audio_", "subs_", 1).rsplit(".", 1)[0] + ".ass"
    )
    if not subtitle_path.exists():
        raise FileNotFoundError(
            f"Missing subtitles for {audio_path.name}: {subtitle_path}. Run 02_assets.py first."
        )

    escaped_path = str(subtitle_path.resolve()).replace("\\", "/")
    escaped_path = escaped_path.replace(":", r"\:").replace("'", r"\'")
    return f"subtitles='{escaped_path}':fontsdir='C\\:/Windows/Fonts'"


def _encoder_args(preview: bool) -> list[str]:
    if preview:
        return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"]
    if SETTINGS.video_encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", "23", "-look_ahead", "0"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]


def _camera_filter(scene, duration: float, width: int, height: int) -> str:
    """Fill the frame, then apply a slow scene-specific camera move."""
    progress = f"(t/{duration:.6f})"
    rhythm_slot = (scene.scene_id - 1) % 3

    if rhythm_slot == 1:
        mode = "drift"
    elif rhythm_slot == 2 or scene.motion.direction == "out":
        mode = "pull"
    else:
        mode = "push"

    if mode == "push":
        scale_factor = f"(1+0.08*{progress})"
        crop_x = f"(in_w-{width})/2 + {progress}*20"
    elif mode == "pull":
        scale_factor = f"(1.08-0.08*{progress})"
        crop_x = f"(in_w-{width})/2"
    else:
        scale_factor = "1.04"
        crop_x = f"(in_w-{width})/2 - 10 + {progress}*20"

    return (
        f"fps=30,scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"scale=w='iw*{scale_factor}':h='ih*{scale_factor}':eval=frame,"
        f"crop={width}:{height}:'{crop_x}':'(in_h-{height})/2'"
    )


def render_scene(
    scene, broll_paths: list[Path], audio_path: Path, duration: float,
    out_dir: Path, width: int, height: int, preview: bool,
) -> Path:
    clip_paths = [broll_paths] if isinstance(broll_paths, Path) else list(broll_paths)
    if not clip_paths:
        raise ValueError(f"Scene {scene.scene_id} has no B-roll clips")

    clip_paths = clip_paths[:2]
    subtitle_filter = _subtitle_filter(audio_path)

    if len(clip_paths) == 1:
        camera_filter = _camera_filter(scene, duration, width, height)
        filter_complex = (
            f"[0:v]{camera_filter},setpts=PTS*1.15,"
            f"tpad=stop_mode=clone:stop_duration={duration:.6f},"
            f"trim=duration={duration:.6f},setpts=PTS-STARTPTS[vbase]"
        )
        audio_index = 1
    else:
        segment_duration = duration / 2
        transition_duration = min(0.5, segment_duration / 2)
        transition_offset = segment_duration - transition_duration
        clip_filters = []
        camera_filter = _camera_filter(scene, segment_duration, width, height)
        for index in range(2):
            clip_filters.append(
            f"[{index}:v]{camera_filter},trim=duration={segment_duration:.6f},"
                f"setpts=PTS-STARTPTS,tpad=stop_mode=clone:"
                f"stop_duration={segment_duration:.6f}[v{index}]"
            )
        filter_complex = ";".join(clip_filters) + (
            f";[v0][v1]xfade=transition=fade:duration={transition_duration:.6f}:"
            f"offset={transition_offset:.6f},"
            f"tpad=stop_mode=clone:stop_duration={transition_duration:.6f},"
            f"trim=duration={duration:.6f},setpts=PTS-STARTPTS[vbase]"
        )
        audio_index = 2

    filter_complex += f";[vbase]{subtitle_filter}[vout]"

    out_path = out_dir / f"scene_{scene.scene_id:03d}.mp4"

    cmd = [
        FFMPEG_BIN, "-y",
    ]
    for clip_path in clip_paths:
        cmd.extend(["-i", str(clip_path)])
    cmd.extend([
        "-i", str(audio_path),
        "-filter_complex", filter_complex,
        "-t", str(duration),
        "-map", "[vout]", "-map", f"{audio_index}:a",
        "-shortest",
        *_encoder_args(preview),
        "-c:a", "aac", "-b:a", "192k",
        "-r", "30",
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
    scenes: list, broll_paths: dict, audio_data: dict, out_dir: Path, preview: bool = False,
) -> list[Path]:
    target_scenes = scenes[:PREVIEW_SCENE_LIMIT] if preview else scenes
    width, height = (PREVIEW_WIDTH, PREVIEW_HEIGHT) if preview else (SETTINGS.width, SETTINGS.height)
    max_workers = min(SETTINGS.max_render_workers, len(target_scenes)) or 1

    results: dict[int, Path] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for scene in target_scenes:
            audio_path, duration = audio_data[scene.scene_id]
            fut = pool.submit(
                render_scene, scene, broll_paths[scene.scene_id], audio_path, duration,
                out_dir, width, height, preview,
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

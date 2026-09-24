"""
Per-scene FFmpeg rendering.

CORRECTION FROM THE PRIOR REVISION: that version used a crop-based zoom
with `eval=frame`, based on documentation for an option that does not
exist in current FFmpeg's crop filter. Tested directly against ffmpeg
6.1.1: `crop`'s w/h expressions cannot reference `t` at all -- it throws
"Error when evaluating the expression" at filter init, every time,
regardless of an `eval` parameter (which crop doesn't accept). crop's x/y
CAN vary with time (confirmed working), but w/h cannot, so a *progressive*
zoom is not achievable with crop on modern FFmpeg.

zoompan is the filter FFmpeg actually provides for this, and it also
tested clean: exact requested duration, real frame-to-frame motion
confirmed by pixel diff, both zoom directions verified. The earlier
concerns about zoompan (variable-framerate source instability) are handled
by normalizing `fps=30` as the FIRST filter in the chain, before zoompan
ever sees the source -- that was always the actual fix, independent of
which zoom filter is used.

drawtext's native `box=1:boxcolor=...:boxborderw=...` (replacing the old
separate drawbox call) is retained from the prior pass -- that fix WAS
verified working, including the "center" position case that used to crash.
"""
from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from config import SETTINGS
import imageio_ffmpeg

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

log = logging.getLogger(__name__)

UPSCALE_FACTOR = 2400
RENDER_TIMEOUT_SEC = 180
PREVIEW_WIDTH, PREVIEW_HEIGHT = 1280, 720
PREVIEW_SCENE_LIMIT = 2


def _escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("%", r"\%")
    )


def _zoompan_filter(direction: str, speed: str, duration: float, width: int, height: int, fps: int) -> str:
    """Progressive Ken Burns zoom. `d` (frames) and the per-frame increment
    are both derived from the scene's actual duration so the zoom always
    completes exactly across the clip regardless of scene length."""
    total_frames = max(1, round(duration * fps))
    zoom_amount = 0.18 if speed == "urgent" else 0.08
    increment = zoom_amount / total_frames

    if direction == "in":
        z_expr = f"min(zoom+{increment:.8f},{1 + zoom_amount})"
    else:
        # start already zoomed in on frame 1, ease back down to 1.0
        z_expr = f"if(eq(on,1),{1 + zoom_amount},max(zoom-{increment:.8f},1.0))"

    return f"zoompan=z='{z_expr}':d={total_frames}:s={width}x{height}:fps={fps}"


def _text_overlay_filter(overlay, duration: float) -> str | None:
    if overlay is None:
        return None

    try:
        if not Path("C:/Windows/Fonts/arial.ttf").exists():
            log.warning("Arial font not found; rendering scene without drawtext overlay")
            return None
    except OSError as error:
        log.warning("Could not check for Arial font (%s); rendering without drawtext overlay", error)
        return None

    font_path = "C\\:/Windows/Fonts/arial.ttf"
    text = _escape_drawtext(overlay.text)
    y_pos = "h-200" if overlay.position == "lower_third" else "(h-text_h)/2"
    alpha_expr = f"if(lt(t\\,0.3)\\,t/0.3\\,if(gt(t\\,{duration}-0.3)\\,({duration}-t)/0.3\\,1))"

    # drawtext's own box=1/boxcolor/boxborderw draw the backing rectangle
    # sized to the actual rendered text within this SAME filter instance --
    # verified working for both lower_third and center positions.
    return (
        f"drawtext=fontfile='{font_path}':text='{text}':fontsize=54:fontcolor=white:"
        f"box=1:boxcolor=black@0.60:boxborderw=20:"
        f"borderw=2:bordercolor=black:x=(w-text_w)/2:y={y_pos}:"
        f"alpha='{alpha_expr}'"
    )


def _encoder_args(preview: bool) -> list[str]:
    if preview:
        return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"]
    if SETTINGS.video_encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", "23", "-look_ahead", "0"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]


def render_scene(
    scene, broll_path: Path, audio_path: Path, duration: float,
    out_dir: Path, width: int, height: int, preview: bool,
) -> Path:
    fps = SETTINGS.fps
    zoom_filter = _zoompan_filter(scene.motion.direction, scene.motion.speed, duration, width, height, fps)
    text_filter = _text_overlay_filter(scene.text_overlay, duration)

    upscale_w = int(width * (UPSCALE_FACTOR / SETTINGS.width))

    # fps normalization MUST come before zoompan -- this is what prevents
    # the variable-framerate-source instability zoompan is usually blamed
    # for; it was never really about the zoom math itself.
    filters = [f"fps={fps}", f"scale={upscale_w}:-1", zoom_filter]
    if text_filter:
        filters.append(text_filter)
    vf_chain = ",".join(filters)

    out_path = out_dir / f"scene_{scene.scene_id:03d}.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(broll_path),
        "-i", str(audio_path),
        "-vf", vf_chain,
        "-t", str(duration),
        "-map", "0:v", "-map", "1:a",
        *_encoder_args(preview),
        "-c:a", "aac", "-b:a", "192k",
        "-r", str(fps),
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

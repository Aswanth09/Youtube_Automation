"""
Step 3: parallel scene rendering + concat + automated mood-based ducked
music bed. Requires 01_plan.py and 02_assets.py to have run first.

Usage:
    python 03_render.py projects/quibi-collapse-ab12cd34 --preview   # fast 720p check, first 2 scenes
    python 03_render.py projects/quibi-collapse-ab12cd34             # full render
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import project_paths, SETTINGS
from project_state import ProjectState
from render_engine import render_all_scenes
from assemble import assemble_final_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("03_render")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: render, concat, music")
    parser.add_argument("project_dir", help="e.g. projects/quibi-collapse-ab12cd34")
    parser.add_argument(
        "--preview", action="store_true",
        help="Render only the first 2 vertical scenes for a fast visual/audio check",
    )
    args = parser.parse_args()

    slug = Path(args.project_dir).name
    paths = project_paths(slug)
    state = ProjectState.load(paths.state_file)

    missing = [
        s.scene_id for s in state.plan.scenes
        if not (
            state.scene_assets.get(s.scene_id)
            and state.scene_assets[s.scene_id].audio_path
            and (
                state.scene_assets[s.scene_id].broll_paths
                or state.scene_assets[s.scene_id].broll_path
            )
        )
    ]
    if missing:
        raise RuntimeError(f"Scenes missing resolved assets: {missing}. Run 02_assets.py first.")

    audio_data = {sid: (Path(sa.audio_path), sa.duration) for sid, sa in state.scene_assets.items()}
    broll_paths = {
        sid: Path(next(path for path in (sa.broll_paths or [sa.broll_path]) if path))
        for sid, sa in state.scene_assets.items()
    }
    subtitles_paths = {}
    for scene in state.plan.scenes:
        assets = state.scene_assets[scene.scene_id]
        subtitle_path = (
            Path(assets.subtitles_path)
            if assets.subtitles_path
            else paths.audio / f"subs_{scene.scene_id:03d}.ass"
        )
        if subtitle_path.exists():
            subtitles_paths[scene.scene_id] = subtitle_path

    scene_count = min(2, len(state.plan.scenes)) if args.preview else len(state.plan.scenes)
    log.info("Rendering %d scene(s) (preview=%s, encoder=%s)...", scene_count, args.preview, SETTINGS.video_encoder)

    scene_files = render_all_scenes(
        state.plan.scenes,
        broll_paths,
        audio_data,
        paths.intermediate_scenes,
        subtitles_paths=subtitles_paths,
        preview=args.preview,
    )

    if args.preview:
        log.info("=== Preview scenes rendered ===")
        for p in scene_files:
            log.info(" -> %s", p)
        log.info("Review these before running the full render (omit --preview).")
        return

    log.info("Concatenating and applying '%s' music bed...", state.plan.suggested_music_mood)
    final_path = assemble_final_video(slug, scene_files, state.plan.suggested_music_mood, paths.final_output)

    state.final_output_path = str(final_path)
    state.save(paths.state_file)

    log.info("=== Build complete: %s ===", final_path)


if __name__ == "__main__":
    main()

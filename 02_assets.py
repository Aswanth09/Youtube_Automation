"""
Step 2: TTS synthesis + B-roll resolution. Reads and updates
project_state.json. Idempotent per scene -- if a scene's audio or B-roll
file already exists on disk, it's skipped rather than re-fetched, so a
failed run can simply be re-run.

Usage:
    python 02_assets.py projects/quibi-collapse-ab12cd34
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import project_paths
from project_state import ProjectState, SceneAssets
from tts_engine import generate_scene_audio
from broll_engine import resolve_all_broll

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("02_assets")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: synthesize audio + resolve B-roll")
    parser.add_argument("project_dir", help="e.g. projects/quibi-collapse-ab12cd34")
    args = parser.parse_args()

    slug = Path(args.project_dir).name
    paths = project_paths(slug)
    state = ProjectState.load(paths.state_file)

    log.info("Synthesizing narration audio for %d scenes...", len(state.plan.scenes))
    for scene in state.plan.scenes:
        existing = state.scene_assets.get(scene.scene_id, SceneAssets())
        subtitle_path = paths.audio / f"subs_{scene.scene_id:03d}.ass"
        if (
            existing.audio_path
            and Path(existing.audio_path).exists()
            and subtitle_path.exists()
        ):
            log.info("Scene %d audio already exists, skipping.", scene.scene_id)
            continue
        audio_path, duration = generate_scene_audio(scene.scene_id, scene.narration, paths.audio)
        existing.audio_path = str(audio_path)
        existing.duration = duration
        state.scene_assets[scene.scene_id] = existing
    state.save(paths.state_file)

    log.info("Resolving B-roll for %d scenes...", len(state.plan.scenes))
    durations = {
        scene_id: assets.duration
        for scene_id, assets in state.scene_assets.items()
    }
    broll_paths = resolve_all_broll(slug, state.plan.scenes, paths.raw_footage, durations)
    for scene_id, paths_for_scene in broll_paths.items():
        existing = state.scene_assets.get(scene_id, SceneAssets())
        existing.broll_paths = [str(path) for path in paths_for_scene]
        existing.broll_path = str(paths_for_scene[0])
        state.scene_assets[scene_id] = existing
    state.save(paths.state_file)

    log.info("=== Assets resolved for projects/%s ===", slug)
    log.info("Next: python 03_render.py projects/%s [--preview]", slug)


if __name__ == "__main__":
    main()

"""
Step 2: TTS synthesis + B-roll resolution. Reads and updates
project_state.json. Idempotent per scene -- if a scene's audio or B-roll
file already exists on disk, it's skipped rather than re-fetched, so a
failed run can simply be re-run.

Usage:
    python 02_assets.py projects/quibi-collapse-ab12cd34
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
if (ROOT_DIR / "pipeline").is_dir() and str(ROOT_DIR / "pipeline") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "pipeline"))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import logging
from pathlib import Path

from pipeline.config import project_paths
from pipeline.project_state import ProjectState, SceneAssets
from pipeline.tts_engine import generate_scene_audio
from pipeline.broll_engine import resolve_all_broll

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("02_assets")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: synthesize audio + resolve B-roll")
    parser.add_argument("project_dir", help="e.g. projects/quibi-collapse-ab12cd34")
    args = parser.parse_args()

    slug = Path(args.project_dir).name
    paths = project_paths(slug)
    state = ProjectState.load(paths.state_file)

    timeline = state.plan.timeline
    log.info("Synthesizing narration audio for %d beats/scenes...", len(timeline))
    for scene in timeline:
        existing = state.scene_assets.get(scene.scene_id, SceneAssets())
        if (
            existing.audio_path
            and Path(existing.audio_path).suffix.lower() == ".wav"
            and Path(existing.audio_path).exists()
            and existing.subtitles_path
            and Path(existing.subtitles_path).exists()
        ):
            log.info("Scene %d audio already exists, skipping.", scene.scene_id)
            continue
        audio_path, ass_path, duration = generate_scene_audio(
            scene.scene_id,
            scene.narration,
            paths.audio,
            speaker=getattr(scene, "speaker", None),
        )
        existing.audio_path = str(audio_path)
        existing.subtitles_path = str(ass_path)
        existing.duration = duration
        state.scene_assets[scene.scene_id] = existing
    state.save(paths.state_file)

    log.info("Resolving B-roll for %d beats/scenes...", len(timeline))
    durations = {
        scene_id: assets.duration
        for scene_id, assets in state.scene_assets.items()
    }
    broll_paths = resolve_all_broll(slug, timeline, paths.raw_footage, durations)
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

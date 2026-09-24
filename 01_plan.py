"""
Step 1: research + structured plan + YouTube metadata. Initializes the
project folder and project_state.json. Does NOT touch TTS, B-roll, or
FFmpeg -- those are 02_assets.py and 03_render.py.

Usage:
    python 01_plan.py "Quibi collapse"
"""
from __future__ import annotations

import argparse
import json
import logging
import re

from config import project_paths
from gemini_engine import stage1_fact_extraction, stage2_generate_scenes
from project_state import new_project_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("01_plan")

WORDS_PER_MINUTE = 150  # rough TTS pacing estimate, used ONLY for the description's timestamp guide


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "video"


def estimate_timestamps(scenes: list) -> list[tuple[int, str]]:
    """Word-count-based estimate (~150 wpm) for the description's
    timestamp section. NOT frame-accurate -- it's built before any audio
    exists. If you want a precise re-cut before publishing, regenerate this
    from project_state.json's measured scene durations after 02_assets.py
    has run."""
    t = 0.0
    marks = []
    for scene in scenes:
        minutes, seconds = int(t // 60), int(t % 60)
        marks.append((scene.scene_id, f"{minutes}:{seconds:02d}"))
        t += (len(scene.narration.split()) / WORDS_PER_MINUTE) * 60
    return marks


def build_final_description(base: str, scenes: list) -> str:
    marks = estimate_timestamps(scenes)
    ts_lines = "\n".join(f"{ts} - Scene {sid}" for sid, ts in marks)
    return f"{base}\n\nTimestamps (approximate, generated from script pacing):\n{ts_lines}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: research + plan + metadata")
    parser.add_argument("topic", help="Company/event to research and script, e.g. 'Quibi collapse'")
    args = parser.parse_args()

    slug = slugify(args.topic)
    paths = project_paths(slug)

    log.info("Stage 1: grounded fact extraction for '%s'...", args.topic)
    fact_brief = stage1_fact_extraction(args.topic)
    log.info("Fact brief captured (%d words).", len(fact_brief.split()))

    log.info("Stage 2: structured scene + metadata generation...")
    plan = stage2_generate_scenes(fact_brief)
    log.info(
        "Generated %d scenes. Music mood: %s. Title: %s",
        len(plan.scenes), plan.suggested_music_mood, plan.metadata.youtube_title,
    )

    state = new_project_state(args.topic, slug, fact_brief, plan)
    state.save(paths.state_file)

    metadata_out = {
        "title": plan.metadata.youtube_title,
        "description": build_final_description(plan.metadata.youtube_description_base, plan.scenes),
        "tags": plan.metadata.youtube_tags,
        "suggested_music_mood": plan.suggested_music_mood,
    }
    paths.metadata_file.write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")

    log.info("=== Project initialized: projects/%s ===", slug)
    log.info("Next: python 02_assets.py projects/%s", slug)


if __name__ == "__main__":
    main()

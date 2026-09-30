from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

import gemini_engine
import music_engine
from schemas import ProjectPlan


class DualHostSchemaTests(unittest.TestCase):
    def test_repeated_speaker_is_rejected(self) -> None:
        plan_data = json.loads(gemini_engine.STAGE2_JSON_EXAMPLE)
        plan_data["beats"][1]["speaker"] = "alice"
        with self.assertRaises(ValidationError):
            ProjectPlan.model_validate(plan_data)

    def test_mocked_gemini_json_populates_alternating_beats(self) -> None:
        response = SimpleNamespace(text=gemini_engine.STAGE2_JSON_EXAMPLE)
        with patch.object(gemini_engine, "_generate_with_fallback", return_value=response):
            plan = gemini_engine._stage2_call("verified sample facts")
        speakers = [beat.speaker for beat in plan.beats or []]
        self.assertEqual(len(speakers), 14)
        self.assertEqual(speakers, ["maya", "jax"] * 7)
        self.assertEqual(plan.music.mood, "tech_panic")


class CuratedMusicTests(unittest.TestCase):
    def test_local_mood_track_and_emergency_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "music_library"
            output_dir = Path(temporary) / "output"
            (root / "tech_panic").mkdir(parents=True)
            (root / "tech_panic" / "panic_fast.mp3").write_bytes(b"curated")
            (root / "fallback.mp3").write_bytes(b"fallback")
            db_path = Path(temporary) / "music.sqlite"

            with (
                patch.object(music_engine, "MUSIC_LIBRARY_DIR", root),
                patch.object(music_engine, "DB_PATH", db_path),
            ):
                selected = music_engine.ensure_music_bed(
                    music_engine.MusicQuery(mood="tech_panic", tempo="fast"),
                    output_dir,
                )
                self.assertTrue(selected.is_file())
                self.assertEqual(selected.read_bytes(), b"curated")

                fallback = music_engine.ensure_music_bed(
                    music_engine.MusicQuery(mood="dark_suspense", tempo="medium"),
                    output_dir,
                )
                self.assertTrue(fallback.is_file())
                self.assertEqual(fallback.read_bytes(), b"fallback")


if __name__ == "__main__":
    unittest.main()

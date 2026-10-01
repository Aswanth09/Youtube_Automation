from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import imageio_ffmpeg
import wave

# pyrefly: ignore [missing-import]
import config
import render_engine
# pyrefly: ignore [missing-import]
import tts_engine
from assemble import build_master_narration_wav
# pyrefly: ignore [missing-import]
from config import AUDIO_CHANNELS, SAMPLE_RATE


class AvatarBootstrapTests(unittest.TestCase):
    def test_missing_default_avatars_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            avatar_dir = Path(temporary)
            maya_path = avatar_dir / "maya_talking.gif"
            jax_path = avatar_dir / "jax_talking.gif"
            maya_idle = avatar_dir / "maya_idle.png"
            jax_idle = avatar_dir / "jax_idle.png"
            with (
                patch.object(config, "AVATAR_DIR", avatar_dir),
                patch.object(config, "AVATAR_MAYA_TALKING", maya_path),
                patch.object(config, "AVATAR_JAX_TALKING", jax_path),
                patch.object(config, "AVATAR_MAYA_IDLE", maya_idle),
                patch.object(config, "AVATAR_JAX_IDLE", jax_idle),
            ):
                maya, jax = config.ensure_default_avatars()

            self.assertTrue(maya.is_file())
            self.assertTrue(jax.is_file())
            self.assertTrue(maya_idle.is_file())
            self.assertTrue(jax_idle.is_file())
            with Image.open(maya) as image:
                maya_image = image.convert("RGBA")
                maya_frames = image.n_frames
                maya_info = dict(image.info)
            with Image.open(jax) as image:
                jax_frames = image.n_frames
                jax_info = dict(image.info)
            self.assertEqual(maya_image.size, (512, 512))
            self.assertEqual(maya_image.getpixel((0, 0))[3], 0)
            self.assertEqual(maya_frames, 4)
            self.assertEqual(jax_frames, 4)
            self.assertEqual(maya_info.get("loop"), 0)
            self.assertEqual(jax_info.get("loop"), 0)
            self.assertEqual(maya_info.get("duration"), 120)
            self.assertEqual(jax_info.get("duration"), 120)


class BeatFiltergraphTests(unittest.TestCase):
    def test_filter_complex_script_parses_with_ffmpeg(self) -> None:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ass_path = root / "subs_001.ass"
            ass_path.write_text(
                """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,5,20,20,20,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:00.40,Default,,0,0,0,,TEST CAPTION
""",
                encoding="utf-8",
            )
            avatar_path = root / "avatar.gif"
            frames = [Image.new("RGBA", (64, 64), color) for color in (
                (255, 0, 255, 255), (240, 0, 240, 255),
                (220, 0, 220, 255), (240, 0, 240, 255),
            )]
            frames[0].save(avatar_path, save_all=True, append_images=frames[1:], duration=120, loop=0)
            beat = SimpleNamespace(beat_id=1, speaker="maya")
            graph = render_engine.build_beat_filtergraph(beat, 0.4, ass_path)
            graph_path = root / "beat.ffgraph"
            graph_path.write_text(graph, encoding="utf-8")

            command = [
                ffmpeg, "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc=size=180x320:rate=30:duration=0.4",
                "-f", "lavfi", "-i", "testsrc2=size=180x320:rate=30:duration=0.4",
                "-ignore_loop", "0", "-i", str(avatar_path),
                "-filter_complex_script", str(graph_path),
                "-map", "[vout]", "-t", "0.4", "-f", "null", "NUL",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=90)
            self.assertEqual(command[command.index("-ignore_loop") + 1], "0")
            self.assertIn("setsar=1", graph)
            self.assertIn("concat=n=2:v=1:a=0", graph)
            self.assertIn("overlay=x=60:y=1320", graph)
            self.assertIn("subtitles=", graph)
            self.assertNotIn("force_style=", graph)

    def test_subtitles_use_center_safe_zone_and_word_karaoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "karaoke.ass"
            boundaries = [
                (0, 4800, "wait", "maya"),
                (4800, 9600, "jax", "maya"),
                (9600, 14400, "really", "maya"),
            ]
            tts_engine._write_ass(path, boundaries)
            content = path.read_text(encoding="utf-8")
            self.assertIn("Arial Black,74", content)
            self.assertIn(",2,40,40,960,1", content)
            self.assertIn(r"{\c&H00D900FF&}WAIT", content)
            self.assertIn(r"{\c&H00FFFFFF&}JAX", content)
            self.assertIn(r"{\c&H00D900FF&}REALLY", content)
            dialogue = [line for line in content.splitlines() if line.startswith("Dialogue:")]
            self.assertEqual(len(dialogue), 3)

    def test_master_narration_inserts_sample_accurate_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            beat_paths = [root / "voice_a.wav", root / "voice_b.wav"]
            for path, value in zip(beat_paths, (b"\x01\x00", b"\x02\x00")):
                with wave.open(str(path), "wb") as audio:
                    audio.setnchannels(AUDIO_CHANNELS)
                    audio.setsampwidth(2)
                    audio.setframerate(SAMPLE_RATE)
                    audio.writeframes(value * 100 * AUDIO_CHANNELS)

            master_path = build_master_narration_wav(beat_paths, root / "master.wav")
            with wave.open(str(master_path), "rb") as master:
                expected_gap_frames = SAMPLE_RATE * config.INTER_TURN_GAP_MS // 1000
                self.assertEqual(master.getframerate(), SAMPLE_RATE)
                self.assertEqual(master.getnchannels(), AUDIO_CHANNELS)
                self.assertEqual(master.getnframes(), 200 + expected_gap_frames)


if __name__ == "__main__":
    unittest.main()

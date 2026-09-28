"""
Central configuration + per-project path resolution.

Global (SETTINGS): API keys, model choice, hardware/render tuning, and the
top-level projects/ and assets/music/ directories.

Per-project (project_paths(slug)): every build gets its own self-contained
workspace under projects/<slug>/ instead of a shared global media/ folder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val or val.startswith("your_"):
        raise RuntimeError(
            f"Missing required environment variable: {key}. "
            f"Copy .env.example to .env and fill in real values."
        )
    return val


VALID_MUSIC_MOODS = ("corporate_tension", "dark_suspense", "slow_investigation")
SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2
VOICE_MAP = {"alice": "en-US-JennyNeural", "bob": "en-US-GuyNeural"}
INTER_TURN_GAP_MS = 120
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
AVATAR_DIR = ASSETS_DIR / "avatars"
AVATAR_ALICE = AVATAR_DIR / "alice.png"
AVATAR_BOB = AVATAR_DIR / "bob.png"
AVATAR_POS_ALICE = (60, 1320)
AVATAR_POS_BOB = (680, 1200)
AVATAR_SIZE = 280


def ensure_default_avatars() -> tuple[Path, Path]:
    """Create circular neon placeholders for missing host avatar images."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError("Pillow is required to generate default host avatars") from error

    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    avatar_specs = (
        (AVATAR_ALICE, "A", (217, 0, 255, 255)),
        (AVATAR_BOB, "B", (0, 212, 255, 255)),
    )
    image_size = 512
    for path, initial, neon in avatar_specs:
        if path.is_file():
            continue
        image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((12, 12, image_size - 12, image_size - 12), fill=(12, 16, 28, 245))
        draw.ellipse((18, 18, image_size - 18, image_size - 18), outline=neon, width=20)
        draw.ellipse((48, 48, image_size - 48, image_size - 48), outline=(*neon[:3], 120), width=4)
        try:
            font = ImageFont.truetype("arial.ttf", 230)
        except OSError:
            font = ImageFont.load_default()
        bounds = draw.textbbox((0, 0), initial, font=font)
        x = (image_size - (bounds[2] - bounds[0])) / 2 - bounds[0]
        y = (image_size - (bounds[3] - bounds[1])) / 2 - bounds[1]
        draw.text((x, y), initial, fill=(255, 255, 255, 255), font=font)
        image.save(path, format="PNG")
    return AVATAR_ALICE, AVATAR_BOB


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    pexels_api_key: str
    gemini_model: str
    tts_voice: str
    max_render_workers: int
    video_encoder: str
    width: int
    height: int
    fps: int
    pexels_max_req_per_hour: int
    projects_dir: Path
    music_dir: Path
    db_path: Path
    stage_transition_pause_sec: float


def load_settings() -> Settings:
    projects_dir = Path(os.getenv("PROJECTS_DIR", "./projects")).resolve()
    projects_dir.mkdir(parents=True, exist_ok=True)

    # SQLite clip library stays centralized (not per-project) -- cross-video
    # dedup requires a shared cache across every project, by design.
    library_dir = projects_dir / "_library"
    library_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        gemini_api_key=_require("GEMINI_API_KEY"),
        pexels_api_key=_require("PEXELS_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        tts_voice=os.getenv("TTS_VOICE", "en-US-ChristopherNeural"),
        max_render_workers=int(os.getenv("MAX_RENDER_WORKERS", "4")),
        video_encoder=os.getenv("VIDEO_ENCODER", "libx264"),
        width=int(os.getenv("OUTPUT_WIDTH", "1080")),
        height=int(os.getenv("OUTPUT_HEIGHT", "1920")),
        fps=int(os.getenv("OUTPUT_FPS", "30")),
        pexels_max_req_per_hour=int(os.getenv("PEXELS_MAX_REQ_PER_HOUR", "180")),
        projects_dir=projects_dir,
        music_dir=Path(os.getenv("MUSIC_DIR", "./assets/music")).resolve(),
        db_path=Path(os.getenv("DB_PATH", str(library_dir / "media_library.db"))).resolve(),
        stage_transition_pause_sec=float(os.getenv("STAGE_TRANSITION_PAUSE_SEC", "3")),
    )


SETTINGS = load_settings()


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def audio(self) -> Path:
        return self.root / "audio"

    @property
    def raw_footage(self) -> Path:
        return self.root / "raw_footage"

    @property
    def intermediate_scenes(self) -> Path:
        return self.root / "intermediate_scenes"

    @property
    def final_output(self) -> Path:
        return self.root / "final_output"

    @property
    def state_file(self) -> Path:
        return self.root / "project_state.json"

    @property
    def metadata_file(self) -> Path:
        return self.root / "metadata.json"

    def ensure(self) -> "ProjectPaths":
        self.root.mkdir(parents=True, exist_ok=True)
        for d in (self.audio, self.raw_footage, self.intermediate_scenes, self.final_output):
            d.mkdir(parents=True, exist_ok=True)
        return self


def project_paths(slug: str) -> ProjectPaths:
    """Every call ensures the folder tree exists -- safe to call from any
    of the 3 entrypoints regardless of which stage created it first."""
    return ProjectPaths(root=SETTINGS.projects_dir / slug).ensure()

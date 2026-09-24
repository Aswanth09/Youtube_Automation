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
        width=int(os.getenv("OUTPUT_WIDTH", "1920")),
        height=int(os.getenv("OUTPUT_HEIGHT", "1080")),
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
        for d in (self.audio, self.raw_footage, self.intermediate_scenes, self.final_output):
            d.mkdir(parents=True, exist_ok=True)
        return self


def project_paths(slug: str) -> ProjectPaths:
    """Every call ensures the folder tree exists -- safe to call from any
    of the 3 entrypoints regardless of which stage created it first."""
    return ProjectPaths(root=SETTINGS.projects_dir / slug).ensure()

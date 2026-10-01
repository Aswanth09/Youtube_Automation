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
VOICE_MAP = {
    "maya": "en-US-JennyNeural",
    "jax": "en-US-GuyNeural",
}
DEFAULT_VOICE = os.getenv("TTS_VOICE", "en-US-ChristopherNeural")
INTER_TURN_GAP_MS = 120
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
AVATAR_DIR = ASSETS_DIR / "avatars"
AVATAR_MAYA_TALKING = ASSETS_DIR / "avatars" / "maya_talking.gif"
AVATAR_MAYA_IDLE = ASSETS_DIR / "avatars" / "maya_idle.png"
AVATAR_JAX_TALKING = ASSETS_DIR / "avatars" / "jax_talking.gif"
AVATAR_JAX_IDLE = ASSETS_DIR / "avatars" / "jax_idle.png"
AVATAR_POS_MAYA = (60, 1320)
AVATAR_POS_JAX = (680, 1200)
AVATAR_SIZE = 280


def ensure_default_avatars() -> tuple[Path, Path]:
    """Create looping neon character avatars and static idle PNGs."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        if AVATAR_MAYA_IDLE.is_file() and AVATAR_JAX_IDLE.is_file():
            return AVATAR_MAYA_TALKING, AVATAR_JAX_TALKING
        raise RuntimeError("Pillow is required to generate default host avatars") from error

    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    avatar_specs = (
        (AVATAR_MAYA_TALKING, AVATAR_MAYA_IDLE, "maya", (217, 0, 255, 255)),
        (AVATAR_JAX_TALKING, AVATAR_JAX_IDLE, "jax", (0, 212, 255, 255)),
    )
    image_size = 512
    mouth_shapes = ((0, 0), (18, 10), (30, 28), (18, 10))

    def _prepare_gif_frame(img: Image.Image) -> Image.Image:
        # Extract alpha mask
        alpha = img.split()[-1]
        # Convert to paletted mode
        p_img = img.convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=255)
        # Set transparent pixels (where alpha < 128) to palette index 255
        mask = Image.eval(alpha, lambda a: 255 if a < 128 else 0)
        p_img.paste(255, mask)
        p_img.info["transparency"] = 255
        return p_img

    def draw_character(kind: str, neon: tuple[int, int, int, int], mouth: tuple[int, int]):
        image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        for inset, alpha, width in ((3, 28, 34), (8, 60, 20), (18, 255, 10)):
            draw.ellipse(
                (inset, inset, image_size - inset, image_size - inset),
                outline=(*neon[:3], alpha),
                width=width,
            )
        draw.ellipse((35, 35, 477, 477), fill=(11, 16, 31, 255))

        if kind == "maya":
            hair = (54, 28, 95, 255)
            draw.ellipse((104, 68, 408, 430), fill=hair)
            draw.polygon([(120, 190), (95, 405), (160, 440), (184, 205)], fill=hair)
            draw.polygon([(365, 175), (425, 400), (360, 445), (334, 195)], fill=hair)
            skin = (245, 190, 165, 255)
            face_box = (143, 105, 369, 389)
        else:
            hair = (37, 39, 54, 255)
            draw.ellipse((115, 69, 397, 410), fill=hair)
            draw.rectangle((136, 140, 376, 360), fill=hair)
            skin = (224, 174, 145, 255)
            face_box = (143, 112, 369, 390)

        draw.ellipse(face_box, fill=skin)
        draw.arc((187, 188, 235, 227), 190, 350, fill=(35, 27, 34, 255), width=8)
        draw.arc((278, 188, 326, 227), 190, 350, fill=(35, 27, 34, 255), width=8)
        draw.ellipse((204, 202, 218, 218), fill=(17, 20, 32, 255))
        draw.ellipse((296, 202, 310, 218), fill=(17, 20, 32, 255))

        if kind == "maya":
            draw.polygon([(135, 176), (153, 92), (238, 72), (212, 138), (180, 158)], fill=hair)
            draw.polygon([(248, 82), (355, 115), (378, 185), (325, 147), (284, 137)], fill=hair)
        else:
            draw.polygon([(133, 158), (145, 99), (207, 69), (295, 78), (372, 125), (377, 164), (327, 132), (269, 145), (210, 127)], fill=hair)
            draw.rounded_rectangle((176, 188, 249, 235), radius=12, outline=(31, 42, 64, 255), width=7)
            draw.rounded_rectangle((270, 188, 343, 235), radius=12, outline=(31, 42, 64, 255), width=7)
            draw.line((249, 208, 270, 208), fill=(31, 42, 64, 255), width=7)

        draw.line((256, 220, 242, 269, 260, 273), fill=(179, 110, 105, 255), width=5)
        mouth_width, mouth_height = mouth
        if mouth_height == 0:
            draw.arc((221, 292, 291, 333), 15, 165, fill=(95, 35, 54, 255), width=7)
        else:
            mouth_box = (256 - mouth_width // 2, 307 - mouth_height // 2,
                         256 + mouth_width // 2, 307 + mouth_height // 2)
            draw.ellipse(mouth_box, fill=(89, 23, 47, 255))
            if mouth_height >= 20:
                draw.rounded_rectangle(
                    (mouth_box[0] + 5, mouth_box[1] + 3, mouth_box[2] - 5, mouth_box[1] + 11),
                    radius=3,
                    fill=(255, 241, 230, 255),
                )
        return image

    for talking_path, idle_path, kind, neon in avatar_specs:
        frames = [draw_character(kind, neon, mouth) for mouth in mouth_shapes]
        if not idle_path.is_file():
            frames[0].save(idle_path, format="PNG")
        if not talking_path.is_file():
            p_frames = [_prepare_gif_frame(f) for f in frames]
            p_frames[0].save(
                talking_path,
                save_all=True,
                append_images=p_frames[1:],
                duration=120,
                loop=0,
                transparency=255,
                disposal=2,
            )

    return (
        AVATAR_MAYA_TALKING,
        AVATAR_JAX_TALKING,
    )


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

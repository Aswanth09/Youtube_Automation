from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

MusicMood = Literal["corporate_tension", "dark_suspense", "slow_investigation"]
Speaker = Literal["maya", "jax"]


class VisualDirection(BaseModel):
    camera_motion: str = Field(default="slow push-in", description="slow push-in | aerial orbit | handheld tracking | static macro")
    graphic_overlay: str = Field(default="none", description="timestamp | redacted document | stat badge | headline ticker | none")
    screen_fx: str = Field(default="clean grade", description="clean grade | subtle film grain | amber tint | night vision")


class Beat(BaseModel):
    beat_id: int = Field(..., ge=1)
    speaker: Speaker
    line: str = Field(...)
    broll_keywords: list[str] = Field(..., min_length=2, max_length=2)
    visual_direction: VisualDirection = Field(default_factory=VisualDirection)

    @field_validator("line")
    @classmethod
    def validate_line_length(cls, value: str) -> str:
        words = len(value.split())
        if not 10 <= words <= 18:
            raise ValueError(f"beat line must contain 10-18 words, got {words}")
        return value

    @field_validator("broll_keywords")
    @classmethod
    def validate_distinct_keywords(cls, value: list[str]) -> list[str]:
        if len({keyword.strip().casefold() for keyword in value}) != len(value):
            raise ValueError("each beat requires two distinct B-roll queries")
        return value

    @property
    def scene_id(self) -> int:
        return self.beat_id

    @property
    def narration(self) -> str:
        return self.line


class MusicQuery(BaseModel):
    mood: Literal["dark_suspense", "corporate_tension", "investigative_fast", "tech_panic"]
    tempo: Literal["medium", "fast"]


class Motion(BaseModel):
    direction: Literal["in", "out"] = "in"
    speed: Literal["calm", "urgent"] = "calm"

    @field_validator("direction", mode="before")
    @classmethod
    def coerce_direction(cls, v: Any) -> str:
        if isinstance(v, str) and "out" in v.lower():
            return "out"
        return "in"

    @field_validator("speed", mode="before")
    @classmethod
    def coerce_speed(cls, v: Any) -> str:
        if isinstance(v, str):
            s = v.lower().strip()
            if s in ("fast", "urgent", "quick", "rapid", "high"):
                return "urgent"
        return "calm"


class TextOverlay(BaseModel):
    text: str = Field(..., max_length=100)
    position: Literal["lower_third", "center"] = "center"


class Scene(BaseModel):
    scene_id: int = Field(..., ge=1, le=7)
    narration: str = Field(...)
    micro_reveal: str = Field(
        ..., min_length=5,
        description="Specific factual detail, number, or plot reveal for this beat",
    )
    pacing_weight: Literal["calm", "urgent"] = "urgent"
    broll_keywords: list[str] = Field(..., min_length=2, max_length=4)
    motion: Motion = Field(default_factory=Motion)
    text_overlay: Optional[TextOverlay] = None
    is_payoff_beat: bool = False

    @field_validator("text_overlay", mode="before")
    @classmethod
    def coerce_text_overlay(cls, v: Any) -> Optional[dict]:
        if not v:
            return None
        if isinstance(v, str):
            clean = v.strip()
            return {"text": clean, "position": "center"} if clean else None
        if isinstance(v, dict) and "text" in v:
            return v
        return None

    @field_validator("pacing_weight", mode="before")
    @classmethod
    def coerce_pacing(cls, v: Any) -> str:
        if isinstance(v, str) and v.lower().strip() in ("fast", "urgent", "high"):
            return "urgent"
        return "calm"

    @field_validator("broll_keywords", mode="before")
    @classmethod
    def coerce_broll(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        if isinstance(v, list):
            return [str(x) for x in v]
        return []

class VideoMetadata(BaseModel):
    youtube_title: str = Field(..., max_length=150)
    youtube_description_base: str
    youtube_tags: list[str] = Field(
        ..., min_length=5, max_length=10,
        description="5 to 10 relevant YouTube tags",
    )

    @field_validator("youtube_tags", mode="before")
    @classmethod
    def coerce_tags(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        if isinstance(v, list):
            return [str(t) for t in v]
        return []


class ProjectPlan(BaseModel):
    beats: list[Beat] | None = Field(default=None, min_length=12, max_length=14)
    music: MusicQuery | None = None
    scenes: list[Scene] | None = Field(default=None, min_length=5, max_length=7)
    suggested_music_mood: MusicMood = "corporate_tension"
    metadata: VideoMetadata

    @field_validator("suggested_music_mood", mode="before")
    @classmethod
    def coerce_mood(cls, v: Any) -> str:
        if isinstance(v, str):
            low = v.lower()
            if "suspense" in low:
                return "dark_suspense"
            if "investig" in low:
                return "slow_investigation"
        return "corporate_tension"

    @model_validator(mode="after")
    def validate_plan_shape(self) -> "ProjectPlan":
        if self.beats is not None:
            if self.music is None:
                raise ValueError("dual-host plans require a music query")
            if not 12 <= len(self.beats) <= 14:
                raise ValueError(f"dual-host plans require 12-14 beats, got {len(self.beats)}")
            if self.beats[0].speaker != "maya":
                raise ValueError("the opening hook beat must be spoken by Maya")
            for previous, current in zip(self.beats, self.beats[1:]):
                if current.speaker == previous.speaker:
                    raise ValueError("dual-host beats must alternate Maya and Jax")
            beat_ids = [beat.beat_id for beat in self.beats]
            if beat_ids != list(range(1, len(self.beats) + 1)):
                raise ValueError("beat_id values must be sequential starting at 1")
            return self

        if self.scenes is None:
            raise ValueError("project plan requires either dual-host beats or legacy scenes")

        total_words = sum(len(scene.narration.split()) for scene in self.scenes)
        if not 145 <= total_words <= 190:
            raise ValueError(f"Short script must contain 145-190 words, got {total_words}")
        for index, scene in enumerate(self.scenes, 1):
            scene.scene_id = index
            word_count = len(scene.narration.split())
            lower, upper = (8, 14) if index == 1 else (20, 35)
            if not lower <= word_count <= upper:
                raise ValueError(
                    f"Scene {index} narration must contain {lower}-{upper} words, got {word_count}"
                )
        return self

    @property
    def timeline(self) -> list[Beat] | list[Scene]:
        return self.beats if self.beats is not None else (self.scenes or [])

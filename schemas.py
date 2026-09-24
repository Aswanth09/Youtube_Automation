from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

MusicMood = Literal["corporate_tension", "dark_suspense", "slow_investigation"]


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
    text: str = Field(..., max_length=250)
    position: Literal["lower_third", "center"] = "lower_third"


class Scene(BaseModel):
    scene_id: int = Field(..., ge=1, le=50)
    narration: str
    micro_reveal: str = Field(default="Key detail")
    pacing_weight: Literal["calm", "urgent"] = "calm"
    broll_keywords: list[str] = Field(default_factory=lambda: ["business", "office"])
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
            return {"text": clean, "position": "lower_third"} if clean else None
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
        return ["business", "office"]

class VideoMetadata(BaseModel):
    youtube_title: str = Field(..., max_length=150)
    youtube_description_base: str
    youtube_tags: list[str] = Field(default_factory=list)

    @field_validator("youtube_tags", mode="before")
    @classmethod
    def coerce_tags(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        if isinstance(v, list):
            return [str(t) for t in v]
        return []


class ProjectPlan(BaseModel):
    scenes: list[Scene] = Field(..., min_length=8, max_length=30)
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

    @field_validator("scenes")
    @classmethod
    def reindex_scenes(cls, v: list[Scene]) -> list[Scene]:
        for i, s in enumerate(v, 1):
            s.scene_id = i
        return v

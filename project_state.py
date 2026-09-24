"""
project_state.json is the single source of truth passed between the 3
standalone entrypoints. Each stage reads it, does its work, and writes it
back -- nothing is held in memory across process boundaries.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from schemas import ProjectPlan


class SceneAssets(BaseModel):
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    broll_paths: list[str] = Field(default_factory=list)
    broll_path: Optional[str] = None
    rendered_scene_path: Optional[str] = None


class ProjectState(BaseModel):
    topic: str
    slug: str
    created_at: str
    fact_brief: str
    plan: ProjectPlan
    scene_assets: dict[int, SceneAssets] = {}
    final_output_path: Optional[str] = None

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProjectState":
        if not path.exists():
            raise FileNotFoundError(
                f"No project_state.json at {path} -- run 01_plan.py first."
            )
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def new_project_state(topic: str, slug: str, fact_brief: str, plan: ProjectPlan) -> ProjectState:
    return ProjectState(
        topic=topic,
        slug=slug,
        created_at=datetime.now(timezone.utc).isoformat(),
        fact_brief=fact_brief,
        plan=plan,
        scene_assets={s.scene_id: SceneAssets() for s in plan.scenes},
    )

"""Two-stage Gemini engine with model failover and structured validation."""
from __future__ import annotations

import json
import logging
import random
import re
import socket
import time
from typing import List, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import ValidationError

from config import SETTINGS
from schemas import ProjectPlan

socket.setdefaulttimeout(15.0)

log = logging.getLogger(__name__)

_client = genai.Client(api_key=SETTINGS.gemini_api_key)
_active_working_model: Optional[str] = None

MODELS_POOL = [
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
]

STAGE1_PROMPT_TEMPLATE = """\
You are a forensic researcher for a business/tech post-mortem documentary series.
Research the following company/event and return ONLY a compact fact-brief in the exact
plain-text format below. Do not write narrative prose, introductions, or conclusions.
Every fact must be verifiable -- if you are not confident in a specific number or date,
omit it rather than guess.

TOPIC: {topic}

Output format (plain text, no markdown, no extra commentary):

KEY_FACTS:
- [fact 1, one sentence, plain and specific]
- [fact 2]
- [fact 3-8, as needed, stop when the core story is covered]

TIMELINE:
- [YYYY-MM or YYYY-MM-DD]: [what happened, one sentence]
- [repeat chronologically, 4-8 entries]

NUMBERS:
- [label]: [figure] (e.g., "Funding raised: $230M", "Peak Valuation: $47B", "Layoffs: 1,200 employees")

KEY_PEOPLE:
- [name] -- [role/title, one sentence on relevance]

RED_FLAGS_AND_REVEALS:
- [a specific overlooked warning sign, ignored audit, internal memo, executive hubris moment, or regulatory red flag -- one sentence each, 4-8 entries]

Keep total output under 600 words.
"""

STAGE2_PROMPT_TEMPLATE = """\
You are a scriptwriter and visual director for a faceless YouTube documentary channel
covering business and tech post-mortems, targeting a US audience. You will generate an
8-minute (approx. 1050-1250 word) video as a structured scene breakdown, plus YouTube metadata.

CRITICAL CONSTRAINT: Use ONLY the facts, dates, numbers, and names provided in the
VERIFIED_FACTS block below. Do not introduce any date, dollar figure, statistic, or named
individual that is not explicitly present in VERIFIED_FACTS.

VERIFIED_FACTS:
{fact_brief}

STRUCTURE:
- Output between 16 and 20 scene objects, with sequential scene_id starting at 1.
- 3-Act structure: Act 1 (scenes ~1-3) catastrophic hook / disaster climax upfront;
  Act 2 (scenes ~4-13) hubris, red flags, warning signs ignored; Act 3 (remaining scenes)
  the unraveling, fallout, and takeaway lesson.

RETENTION REQUIREMENTS:
- Scene 1 is an EXTREME COLD OPEN: open on the catastrophic collapse, the vanished
  valuation, or the shocking end-state contrast. Do NOT name the company in scene 1 --
  hook on the "what happened" before revealing the "who". The company name can be introduced starting scene 2.
- Every scene's `micro_reveal` field must name one specific item from VERIFIED_FACTS'
  RED_FLAGS_AND_REVEALS section (or a specific fact/number if reveals are exhausted).
- Every scene's narration must end on a forward-tension bridge into the next beat (an
  open question, an ominous foreshadow, an unresolved thread) rather than a flat,
  concluded statement -- EXCEPT the final scene, which resolves with the takeaway lesson.
- Assign pacing_weight "urgent" to the hook, the climax, and payoff-beat scenes;
  "calm" to context-building scenes.
- Mark 2-3 scenes as is_payoff_beat: true -- the story's biggest reveals or turning points.
- Each narration should be 25-35 words.
- For each scene, generate 2-3 ranked broll_keywords appropriate for stock footage search (e.g. "empty office chairs").
- Only add a text_overlay for payoff-beat scenes or scenes containing a specific number from VERIFIED_FACTS.

MUSIC:
- Set `suggested_music_mood` to exactly one of: "corporate_tension", "dark_suspense", "slow_investigation".

METADATA:
- `youtube_title`: under 100 characters, curiosity-driven, accurate to the story.
- `youtube_description_base`: exactly 3 paragraphs of SEO-optimized description text (omit timestamps).
- `youtube_tags`: exactly 15 tags targeted at US search behavior.
"""

STAGE2_JSON_EXAMPLE = """\
Return one JSON object shaped like this (the real response must contain 16-20 scenes):
{
    "scenes": [{
        "scene_id": 1,
        "narration": "The collapse arrived before anyone admitted the warning signs were real, but the next revelation explains why the damage spread so quickly.",
        "micro_reveal": "A specific warning sign from RED_FLAGS_AND_REVEALS",
        "pacing_weight": "urgent",
        "broll_keywords": ["empty office", "falling stock chart"],
        "motion": {"direction": "in", "speed": "urgent"},
        "text_overlay": null,
        "is_payoff_beat": false
    }],
    "suggested_music_mood": "corporate_tension",
    "metadata": {
        "youtube_title": "A curiosity-driven title under 100 characters",
        "youtube_description_base": "First SEO paragraph.\n\nSecond SEO paragraph.\n\nThird SEO paragraph.",
        "youtube_tags": ["tag01", "tag02", "tag03", "tag04", "tag05", "tag06", "tag07", "tag08", "tag09", "tag10", "tag11", "tag12", "tag13", "tag14", "tag15"]
    }
}
"""


def _get_candidate_models() -> List[str]:
    """Return active, configured, and fallback models in deduplicated order."""
    models = [_active_working_model, *MODELS_POOL, SETTINGS.gemini_model]
    seen: set[str] = set()
    candidates: List[str] = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            candidates.append(model)
    return candidates


def _generate_with_fallback(
    contents: str,
    config: Optional[types.GenerateContentConfig] = None,
    max_cycles: int = 3,
) -> types.GenerateContentResponse:
    """Generate content while cycling through available models after failures."""
    if max_cycles < 1:
        raise ValueError("max_cycles must be at least 1")

    candidates = _get_candidate_models()
    last_error: Optional[Exception] = None

    for cycle in range(1, max_cycles + 1):
        for model_name in candidates:
            try:
                log.info("Requesting generation from model '%s'", model_name)
                response = _client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                global _active_working_model
                _active_working_model = model_name
                return response
            except (APIError, TimeoutError, Exception) as error:
                last_error = error
                log.warning(
                    "Model '%s' failed with %s: %s; failing over",
                    model_name,
                    type(error).__name__,
                    error,
                )
                time.sleep(1.0)

        if cycle < max_cycles:
            backoff = 2.0 * cycle
            log.warning(
                "All candidate models failed on cycle %d/%d; retrying in %.1fs",
                cycle,
                max_cycles,
                backoff,
            )
            time.sleep(backoff)

    raise RuntimeError(
        f"All candidate models exhausted after {max_cycles} cycles: {last_error}"
    ) from last_error


def stage1_fact_extraction(topic: str) -> str:
    """Extract a plain-text fact brief without using Google Search tools."""
    prompt = STAGE1_PROMPT_TEMPLATE.format(topic=topic)
    response = _generate_with_fallback(prompt)
    fact_brief = (response.text or "").strip()
    if len(fact_brief.split()) < 30:
        raise ValueError("Stage 1 fact brief suspiciously short -- empty response.")
    return fact_brief


def _strip_json_fences(text: str) -> str:
    """Remove an optional markdown JSON fence from a model response."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    return cleaned.strip()


def _stage2_call(fact_brief: str) -> ProjectPlan:
    prompt = (
        f"{STAGE2_PROMPT_TEMPLATE.format(fact_brief=fact_brief)}\n\n"
        f"You MUST return valid JSON matching this compact example and all field constraints:\n"
        f"{STAGE2_JSON_EXAMPLE}"
    )
    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
    except Exception as error:
        log.warning(
            "Thinking configuration is unavailable in this SDK (%s); using standard generation config",
            error,
        )
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=4096,
        )
    response = _generate_with_fallback(prompt, config=config)
    raw_text = (response.text or "").strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = raw_text[start:end + 1]
    else:
        json_str = raw_text

    data = json.loads(json_str, strict=False)
    return ProjectPlan.model_validate(data)


def _pause_between_stages() -> None:
    """Pause between generation stages to avoid burst limits."""
    delay = SETTINGS.stage_transition_pause_sec + random.uniform(0.5, 1.5)
    log.info("Pausing %.1fs between Stage 1 and Stage 2", delay)
    time.sleep(delay)


def stage2_generate_scenes(fact_brief: str, max_attempts: int = 3) -> ProjectPlan:
    """Generate scenes with self-healing retries for schema validation failures."""
    _pause_between_stages()
    last_error: Optional[Exception] = None
    prompt_fact_brief = fact_brief

    for attempt in range(1, max_attempts + 1):
        try:
            plan = _stage2_call(prompt_fact_brief)
            verify_scenes_against_facts(plan, fact_brief)
            return plan
        except (ValidationError, ValueError) as error:
            last_error = error
            log.warning(
                "Stage 2 attempt %d/%d failed validation: %s",
                attempt,
                max_attempts,
                error,
            )
            prompt_fact_brief = (
                f"{fact_brief}\n\n"
                f"[SYSTEM NOTE: your previous attempt failed schema validation: {error}. Fix it.]"
            )

    raise RuntimeError(
        f"Stage 2 failed after {max_attempts} attempts: {last_error}"
    ) from last_error


def verify_scenes_against_facts(plan: ProjectPlan, fact_brief: str) -> None:
    """Warn when scene narration contains a year or currency figure absent from facts."""
    brief_lower = fact_brief.lower()
    number_pattern = re.compile(r"\$[\d,.]+[mMbBkK]?|\b(?:19|20)\d{2}\b")

    for scene in plan.scenes:
        for token in number_pattern.findall(scene.narration):
            if token.lower() not in brief_lower:
                log.warning(
                    "Scene %d mentions figure/year %r not found in fact brief.",
                    scene.scene_id,
                    token,
                )

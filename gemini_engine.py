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
    "gemini-3.1-flash-lite",
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
You are the lead writer and visual director for a US YouTube Shorts channel. Write a
65-72 second psychological thriller in the high-tension documentary style of
MagnatesMedia and ColdFusion: visceral, paranoid, cinematic, and built to replay.
Every scene must escalate the investigation and make the next cut feel unavoidable.

CRITICAL CONSTRAINT: Use ONLY the facts, dates, numbers, and names provided in the
VERIFIED_FACTS block below. Do not introduce any date, dollar figure, statistic, or named
individual that is not explicitly present in VERIFIED_FACTS.

VERIFIED_FACTS:
{fact_brief}

STRUCTURE:
- Output exactly 6 scene objects with sequential scene_id starting at 1.
- STRICT LENGTH: 150-180 words total across 6 scenes (~60-70 seconds runtime). Scene 1
    is the 0-3 second hook; scenes 2-5 accelerate through hubris, deception, secret
    backdoors, and the sudden 72-hour bank run; scene 6 is the payoff and loop.

RETENTION REQUIREMENTS:
- Scene 1 must contain exactly 10-14 words. Open with a visceral visual metaphor and
    dollar loss before revealing the entity name. Withhold the name until scene 2 and
    include a centered stat-card overlay such as "$32,000,000,000 -> ZERO".
- Scenes 2-5 should target 25-30 words each, using fast escalation through hubris,
    deception, secret backdoors, hidden incentives, and the sudden 72-hour bank run.
    Avoid dry reporting and repetitive "In [date], X" sentence openings.
- Scene 6 must echo the opening hook's metric or question to create an endless loop and
    must end with this exact sentence: "Follow for the next collapse."
- Every scene's `micro_reveal` must name one specific verified fact, red flag, or figure.
- Use "urgent" pacing by default. Return exactly 2 or 3 distinct, ranked, motion-heavy
    B-roll queries per scene.
- B-ROLL KEYWORDS RULES:
    Strictly avoid abstract lifestyle metaphors (NO sports cars, NO champagne, NO nightclubs).
    Output ONLY concrete, high-tension financial documentary and white-collar drama keywords.
    Use exact visual anchors like:
    - "red stock market graph falling"
    - "trader hands on head stressed"
    - "counting hundred dollar bills fast"
    - "server rack blinking dark room"
    - "financial audit documents rubber stamp"
    - "handcuffs police white collar crime"
    - "closing office glass doors night"
    - "empty trading floor after hours"
- Set `is_payoff_beat` true for major reveal scenes and use centered stat-card overlays
    when a verified figure needs visual emphasis.

MUSIC:
- Set `suggested_music_mood` to exactly one of: "corporate_tension", "dark_suspense", "slow_investigation".

METADATA:
- `youtube_title`: under 100 characters, curiosity-driven, accurate to the story.
- `youtube_description_base`: 2-3 paragraphs of SEO-optimized description text (omit timestamps).
- `youtube_tags`: 5-10 tags targeted at US search behavior, including `#Shorts`.
"""

STAGE2_JSON_EXAMPLE = """\
{
    "scenes": [
        {
            "scene_id": 1,
            "narration": "Forty-seven billion dollars vanished. Then, ninety days later, absolutely nothing remained.",
            "micro_reveal": "Peak valuation vs total collapse",
            "pacing_weight": "urgent",
            "broll_keywords": ["empty office lobby", "shattered glass falling", "dark skyscrapers night"],
            "motion": {"direction": "in", "speed": "urgent"},
            "text_overlay": {"text": "$47,000,000,000 → $0", "position": "center"},
            "is_payoff_beat": true
        },
        {
            "scene_id": 2,
            "narration": "This was not a standard market crash. This was WeWork, an empire built on charismatic storytelling that seduced the smartest venture capitalists on Earth while everyone applauded the illusion.",
            "micro_reveal": "WeWork communal workspace movement pitch",
            "pacing_weight": "urgent",
            "broll_keywords": ["modern startup office", "investor meeting room", "city skyline time lapse"],
            "motion": {"direction": "in", "speed": "urgent"},
            "text_overlay": null,
            "is_payoff_beat": false
        },
        {
            "scene_id": 3,
            "narration": "SoftBank alone pumped over ten billion dollars into the furnace, pricing an ordinary real estate subleasing company higher than the world's largest commercial airlines while risk warnings disappeared behind euphemisms.",
            "micro_reveal": "SoftBank massive capital infusions and high valuation",
            "pacing_weight": "urgent",
            "broll_keywords": ["currency stacks", "luxury penthouse dusk", "financial stock chart falling"],
            "motion": {"direction": "in", "speed": "urgent"},
            "text_overlay": {"text": "$10B Invested", "position": "center"},
            "is_payoff_beat": false
        },
        {
            "scene_id": 4,
            "narration": "Behind closed doors, Adam Neumann trademarked the common word 'We' and charged his own cash-strapped company six million dollars just to use it while investors were told it represented culture.",
            "micro_reveal": "Company paid Neumann $6 million for the 'We' trademark",
            "pacing_weight": "urgent",
            "broll_keywords": ["legal contract signing", "money transfer digital", "boardroom argument"],
            "motion": {"direction": "in", "speed": "urgent"},
            "text_overlay": {"text": "$6M Trademark Payout", "position": "center"},
            "is_payoff_beat": true
        },
        {
            "scene_id": 5,
            "narration": "Even worse, he took personal loans against company stock to purchase commercial buildings, then leased those exact properties right back to WeWork for massive private profit, and nobody stopped the transaction.",
            "micro_reveal": "2019 S-1 filing revealed massive lease liabilities",
            "pacing_weight": "urgent",
            "broll_keywords": ["red financial spreadsheet", "anxious investor phone call", "courtroom gavel"],
            "motion": {"direction": "in", "speed": "urgent"},
            "text_overlay": {"text": "-$2B Annual Loss", "position": "center"},
            "is_payoff_beat": false
        },
        {
            "scene_id": 6,
            "narration": "The S-1 filing revealed two billion in losses and forty-seven billion in lease liabilities. Forty-seven billion erased in weeks before the lights went out completely in one brutal weekend. Follow for the next collapse.",
            "micro_reveal": "Chapter 11 bankruptcy filing and final collapse",
            "pacing_weight": "urgent",
            "broll_keywords": ["closing office doors", "empty corporate building night", "bankruptcy sign"],
            "motion": {"direction": "out", "speed": "urgent"},
            "text_overlay": {"text": "Follow for more", "position": "center"},
            "is_payoff_beat": true
        }
    ],
    "suggested_music_mood": "dark_suspense",
    "metadata": {
        "youtube_title": "The $47 Billion Lie: How WeWork Burned It All #Shorts",
        "youtube_description_base": "How Adam Neumann burned $47 billion in 90 days. The truth behind the WeWork collapse.\\n\\nSubscribe for more corporate downfalls and financial post-mortems.",
        "youtube_tags": ["#Shorts", "WeWork", "Adam Neumann", "Business Documentary", "Startup Failure", "Tech Collapse"]
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
    plan = ProjectPlan.model_validate(data)
    _validate_stage2_constraints(plan)
    return plan


def _validate_stage2_constraints(plan: ProjectPlan) -> None:
    """Reject weak new YouTube Short plans during generation."""
    if not 5 <= len(plan.scenes) <= 7:
        raise ValueError(f"Short plan must contain 5-7 scenes, got {len(plan.scenes)}")

    total_words = sum(len(scene.narration.split()) for scene in plan.scenes)
    if not 145 <= total_words <= 190:
        raise ValueError(f"Short script must contain 145-190 words, got {total_words}")

    for scene in plan.scenes:
        word_count = len(scene.narration.split())
        lower, upper = (8, 14) if scene.scene_id == 1 else (20, 35)
        if not lower <= word_count <= upper:
            raise ValueError(
                f"Scene {scene.scene_id} narration must contain {lower}-{upper} words, got {word_count}"
            )

        keywords = [keyword.strip() for keyword in scene.broll_keywords if keyword.strip()]
        if not 2 <= len(keywords) <= 3:
            raise ValueError(
                f"Scene {scene.scene_id} must contain 2-3 B-roll keywords, got {len(keywords)}"
            )
        if len({keyword.casefold() for keyword in keywords}) != len(keywords):
            raise ValueError(f"Scene {scene.scene_id} B-roll keywords must be distinct")

    if plan.scenes[0].text_overlay is None:
        raise ValueError("Scene 1 must include a centered stat-card text_overlay")
    if not plan.scenes[-1].narration.endswith("Follow for the next collapse."):
        raise ValueError('Final scene must end with "Follow for the next collapse."')


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

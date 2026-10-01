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

STAGE2_PROMPT_TEMPLATE = """
You are an elite short-form video dialogue writer for 60-70 second investigative thrillers on YouTube Shorts and TikTok.
Create a high-retention 12-14 beat dialogue script grounded ONLY in VERIFIED_FACTS. Return one JSON object matching ProjectPlan: `beats`, `music`, and `metadata`.

TOPIC TITLE / SPECIFIC ANGLE:
{topic_title}

VERIFIED_FACTS:
{fact_brief}

NARRATIVE ARCHETYPE (Choose the best fit for the topic):

* Engine A (Hidden Failure): Overlooked technical/product flaw -> cascading failure -> original decision reframed.
* Engine B (Corporate Contradiction): Public claims vs. secret reality -> evidence leak -> documented fallout.
* Engine C (Digital Contagion): Private chat/warning -> network panic -> instantaneous bank run or liquidity drain.

DUAL-HOST FORMAT & ROLES:

* Maya: Skeptical, inquisitive interrogator. Speaks for viewer disbelief, calls out contradictions, and drives curiosity.
* Jax: Calm, authoritative forensic analyst. Delivers exact verified figures, dates, leaked memos, and causal explanations.
* Alternate speakers on every single turn: Maya, Jax, Maya, Jax... Start with Maya.

BEAT-BY-BEAT RETENTION STRUCTURE (14 TURNS):

* Beat 1 (Maya): High-tension pattern interrupt hook. MUST address Jax directly by name about the specific premise.
* Beat 2 (Jax): Opening shock statistic or verified fact. MUST address Maya directly by name.
* Beats 3-6 (Genesis): Establish the mechanism and the first overlooked red flag.
* Beats 7-10 (Escalation): Reveal the hidden leak, conflict of interest, or accelerating contagion.
* Beats 11-12 (Cascade): The irreversible event, peak withdrawal/loss numbers, and regulator seizure.
* Beat 13 (Maya): Setup for the paradoxical loop ("Jax, then the company didn't collapse when...").
* Beat 14 (Jax): The punchline reframing Beat 1 so the video loops seamlessly back to the beginning.

CRITICAL CONSTRAINTS:

* Each `line` MUST contain strictly 10-18 words. No rambling, no dead air, no academic lectures.
* Use 12-14 sequential `beat_id` values (target 14).
* Never invent figures, memos, or dates not grounded in the facts.

CRITICAL B-ROLL SEARCH RULES (Universal Stock Matching):
Each beat must specify exactly two distinct, concrete visual queries formatted as: `[Physical Subject] + [Action / Environment]` (strictly 2 to 4 words).
- STOCK FOOTAGE REALITY: Pexels indexes real-world filmed footage, NOT abstract concepts, proprietary campus names, or UI dialog boxes.
- FORBIDDEN:
  * Never use proprietary trademarks or campus names (e.g., 'Googleplex', 'Apple Park', 'OpenAI HQ', 'Kremlin room'). Use visual proxies: 'modern glass corporate office', 'high tech headquarters building', 'government conference room'.
  * Never use abstract UI/software states (e.g., 'error popup dialog', 'code red alert', 'ai thinking'). Use physical human/hardware proxies: 'frustrated programmer staring at monitor', 'blinking server rack lights', 'hands typing urgently on keyboard'.
  * Never use isolated generic nouns that match facilities (e.g., 'sign', 'entrance', 'door', 'window', 'board'). Always qualify: 'corporate glass exterior', 'office hallway corridor', 'cybersecurity operations center'.
- GENRE EXAMPLES:
  * AI & Tech: 'glowing data center servers', 'robotics lab engineer', 'futuristic microchip macro'
  * Military / Geopolitics: 'military command center screens', 'satellite map radar', 'press briefing microphone podium'
  * Crime / Investigation: 'detective reviewing evidence board', 'police tape flashing sirens', 'courtroom judge gavel'
  * Business & Markets: 'stock market exchange trading floor', 'empty skyscraper boardroom', 'financial charts digital screen'

MUSIC:

* Return `music` with mood one of `dark_suspense`, `corporate_tension`, `investigative_fast`, `tech_panic` and tempo `medium` or `fast`.

METADATA:

* `youtube_title`: under 100 characters and viral/accurate.
* `youtube_description_base`: 2 concise paragraphs with context.
* `youtube_tags`: 5-10 relevant tags including `#Shorts`.
"""

STAGE2_JSON_EXAMPLE = """
{
"beats": [
{"beat_id": 1, "speaker": "maya", "line": "Wait Jax, did one private message thread really trigger forty-two billion dollars in withdrawals?", "broll_keywords": ["smartphone notification screen", "panicked person holding phone"]},
{"beat_id": 2, "speaker": "jax", "line": "Maya, once venture capital leaders sent warnings, the digital bank run became completely instantaneous.", "broll_keywords": ["message notification popping up", "venture capital office interior"]},
{"beat_id": 3, "speaker": "maya", "line": "How did startup founders bring down a two hundred billion dollar bank in hours?", "broll_keywords": ["bank headquarters exterior sign", "graph showing sharp financial decline"]},
{"beat_id": 4, "speaker": "jax", "line": "Their growth depended on customer deposits remaining safe while billions sat in long-term bonds.", "broll_keywords": ["bank deposit ledger pages", "interest rate chart moving upwards"]},
{"beat_id": 5, "speaker": "maya", "line": "Somebody inside had to notice rising interest rates were crushing those bond investments, right?", "broll_keywords": ["financial audit report highlighted", "empty executive chair boardroom"]},
{"beat_id": 6, "speaker": "jax", "line": "The bank operated without a chief risk officer for eight months during interest hikes.", "broll_keywords": ["empty office chair", "stock trading dashboard flashing red"]},
{"beat_id": 7, "speaker": "maya", "line": "Wait, who actually knew eighty-nine percent of all customer deposits were completely uninsured?", "broll_keywords": ["scrolling through chat messages", "stressed founder looking at laptop"]},
{"beat_id": 8, "speaker": "jax", "line": "Prominent venture funds realized the exposure and advised portfolio companies to withdraw immediately.", "broll_keywords": ["mobile banking withdrawal screen", "hand hitting send on smartphone"]},
{"beat_id": 9, "speaker": "maya", "line": "So private screenshots leaked while executives were publicly promising everything was totally fine?", "broll_keywords": ["private founder chat messages", "chaotic crowd of tech employees"]},
{"beat_id": 10, "speaker": "jax", "line": "Customers withdrew forty-two billion in ten hours, draining one quarter of total deposits.", "broll_keywords": ["cash vault door slamming shut", "empty trading floor after hours"]},
{"beat_id": 11, "speaker": "maya", "line": "Forty-two billion gone in ten hours? And leadership couldn't halt the mobile withdrawals?", "broll_keywords": ["panicked trader hands on head", "regulator notice on bank doors"]},
{"beat_id": 12, "speaker": "jax", "line": "Regulators seized the insolvent bank within forty-eight hours of their failed capital raise.", "broll_keywords": ["FDIC sign on bank door", "closing office glass doors night"]},
{"beat_id": 13, "speaker": "maya", "line": "Jax, then the bank didn't die from bad bonds. It collapsed when...", "broll_keywords": ["smartphone notification screen", "bank headquarters exterior sign"]},
{"beat_id": 14, "speaker": "jax", "line": "...digital group chats turned mutual trust into the fastest bank run in history.", "broll_keywords": ["message notification popping up", "panicked person holding phone"]}
],
"music": {"mood": "tech_panic", "tempo": "fast"},
"metadata": {
"youtube_title": "Inside the WhatsApp Panic That Killed Silicon Valley Bank #Shorts",
"youtube_description_base": "How a private message thread sparked a $42 billion bank run in under 48 hours.\\n\\nFollow the warning signs behind the collapse.",
"youtube_tags": ["#Shorts", "SVB", "BankRun", "FinancialCrisis", "SiliconValleyBank", "TechPanic"]
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


def _stage2_call(fact_brief: str, topic_title: str | None = None) -> ProjectPlan:
    prompt = (
        f"{STAGE2_PROMPT_TEMPLATE.format(fact_brief=fact_brief, topic_title=topic_title or 'No topic title supplied')}\n\n"
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
    """Apply generation constraints to dual-host beats and legacy scene plans."""
    if plan.beats is not None:
        if not 12 <= len(plan.beats) <= 14:
            raise ValueError(f"Dual-host Short requires 12-14 beats, got {len(plan.beats)}")
        if plan.beats[0].speaker != "maya":
            raise ValueError("The first hook beat must be spoken by Maya")
        if not re.search(r"\bjax\b", plan.beats[0].line, flags=re.IGNORECASE):
            raise ValueError("Beat 1 must address Jax by name")
        if not re.search(r"\bmaya\b", plan.beats[1].line, flags=re.IGNORECASE):
            raise ValueError("Beat 2 must address Maya by name")
        for previous, current in zip(plan.beats, plan.beats[1:]):
            if previous.speaker == current.speaker:
                raise ValueError("Beat speakers must alternate between Maya and Jax")
            word_count = len(current.line.split())
            if not 10 <= word_count <= 18:
                raise ValueError(
                    f"Beat {current.beat_id} line must contain 10-18 words, got {word_count}"
                )
            if len(current.broll_keywords) != 2:
                raise ValueError(f"Beat {current.beat_id} must have exactly 2 B-roll queries")
        return

    if plan.scenes is None:
        raise ValueError("Plan has no scenes or dialogue beats")
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
            raise ValueError(f"Scene {scene.scene_id} must contain 2-3 B-roll keywords, got {len(keywords)}")
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


def stage2_generate_scenes(
    fact_brief: str,
    max_attempts: int = 3,
    topic_title: str | None = None,
) -> ProjectPlan:
    """Generate scenes with self-healing retries for schema validation failures."""
    _pause_between_stages()
    last_error: Optional[Exception] = None
    prompt_fact_brief = fact_brief

    for attempt in range(1, max_attempts + 1):
        try:
            plan = _stage2_call(prompt_fact_brief, topic_title=topic_title)
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
    """Warn when narration contains a year or currency figure absent from facts."""
    brief_lower = fact_brief.lower()
    number_pattern = re.compile(r"\$[\d,.]+[mMbBkK]?|\b(?:19|20)\d{2}\b")

    for beat in plan.timeline:
        narration = beat.line if hasattr(beat, "line") else beat.narration
        beat_id = beat.beat_id if hasattr(beat, "beat_id") else beat.scene_id
        for token in number_pattern.findall(narration):
            if token.lower() not in brief_lower:
                log.warning(
                    "Beat/scene %d mentions figure/year %r not found in fact brief.",
                    beat_id,
                    token,
                )

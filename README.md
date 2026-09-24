# Automated Post-Mortem Documentary Pipeline (v2)

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # then fill in GEMINI_API_KEY and PEXELS_API_KEY
```

Requires `ffmpeg` and `ffprobe` on PATH. Check Quick Sync support with:
```bash
ffmpeg -encoders | findstr qsv
```
Set `VIDEO_ENCODER=h264_qsv` in `.env` if supported (faster, lower heat on
Intel iGPUs like Iris Xe); otherwise leave `libx264`.

Add 3 royalty-free instrumental tracks to `assets/music/` -- see
`assets/music/README.md` for the exact required filenames.

## Run — 3 decoupled steps

Each step is a standalone process communicating through
`projects/<slug>/project_state.json`, so a 30+ minute build no longer has
to happen in one blocking `main.py` run. Re-running a step is safe; it
picks up from whatever's already on disk.

```bash
# Step 1 -- research, script, metadata. Creates projects/<slug>/
python 01_plan.py "Quibi collapse"

# Step 2 -- TTS + B-roll. Idempotent per scene.
python 02_assets.py projects/quibi-collapse-ab12cd34

# Step 3 -- render + concat + music. Check with --preview first.
python 03_render.py projects/quibi-collapse-ab12cd34 --preview
python 03_render.py projects/quibi-collapse-ab12cd34
```

Final video: `projects/<slug>/final_output/<slug>_final.mp4`
YouTube metadata (title/description/tags): `projects/<slug>/metadata.json`

## Project folder layout

```
projects/<topic_slug>/
  project_state.json       # shared state passed between all 3 steps
  metadata.json             # YouTube title, description w/ timestamps, tags
  audio/                     # per-scene TTS files
  raw_footage/               # downloaded Pexels B-roll for this project
  intermediate_scenes/       # rendered per-scene mp4 chunks
  final_output/               # <slug>_final.mp4
projects/_library/
  media_library.db           # SHARED SQLite cache + cross-project dedup log
```

## Notes on this build

- `GEMINI_MODEL` defaults to `gemini-2.5-pro`. `01_plan.py` pauses
  (`STAGE_TRANSITION_PAUSE_SEC` + jitter) between Stage 1 and Stage 2 on
  top of per-call retry/backoff, since that handoff is the most common
  place to trip a burst rate limit.
- Each scene's `micro_reveal` field is a structural requirement, not just a
  prompt suggestion -- Pydantic rejects a Stage 2 response missing it,
  which is what enforces the "every scene exposes a specific unrevealed
  truth" retention rule from the schema level, not just wording.
- Timestamps in `metadata.json`'s description are estimated from narration
  word count (~150 wpm) at plan time, before any audio exists -- they're a
  publishing aid, not frame-accurate. Re-derive from `project_state.json`'s
  measured `scene_assets[*].duration` after Step 2 if you want precision
  before publishing.
- B-roll resolution (`broll_engine.py`) is intentionally sequential and
  runs *before* the parallel render phase -- do not move it inside
  `render_engine`'s worker pool.
- `MAX_RENDER_WORKERS` defaults to 4 -- appropriate for a U-series laptop
  CPU (e.g. i7-1355U). `--preview` further caps this to the 2 preview
  scenes regardless of the setting.
- `projects/` and `.env` are git-ignored. `assets/music/` is intentionally
  committed -- it's a static dependency, not generated output.

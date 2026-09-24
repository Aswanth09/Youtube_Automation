# Required music assets

`03_render.py` automatically loads a track based on Gemini's
`suggested_music_mood` field -- no manual `--music` flag needed. Place 3
royalty-free instrumental tracks here, named exactly:

- `corporate_tension.mp3`
- `dark_suspense.mp3`
- `slow_investigation.mp3`

Any royalty-free source works (e.g. YouTube Audio Library, Pixabay Music) --
just verify each track's license permits monetized use before publishing.
Claude cannot provide actual audio files here; these three filenames are
the contract `assemble.py`'s `MOOD_TRACK_MAP` expects.

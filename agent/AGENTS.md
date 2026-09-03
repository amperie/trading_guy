# Agent instructions

Canonical instructions live in `agent/`. Read them before making changes.

1. Open `agent/INDEX.md` and load only the topic files that match this task.
2. Follow `agent/authoring.md` when adding or editing instruction files.
3. Do not override methods marked `@final`.
4. Instantiate components from config via `utils.utils.instantiate_from_string()`.
5. Put new code in the directories listed in `agent/layout.md`.
6. Check `agent/TODO.md` for related follow-ups.

`agent/INDEX.md` catalogues instruction files and code locations.
`agent/TODO.md` is the standing work list; it is not a 30-line instruction file.

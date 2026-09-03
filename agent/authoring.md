# Writing agent instructions

Keep each instruction file in `agent/` to **30 lines or less** (including headings). One concern per file.

Do not put standing work in instruction files. Put it in `agent/TODO.md` (no line cap).

When adding a file:
1. Create `agent/<topic>.md` with a single H1 and only the rules needed for that topic.
2. Add a row to `agent/INDEX.md` (path, when to read, related code).
3. Link it from `agent/AGENTS.md` only if it is required on every task.
4. Use repo paths, class names, and `@final` notes. Do not restate other files.

When editing:
- Trim instead of appending. Split if a file would exceed 30 lines.
- Update INDEX if the purpose, path, or related code changed.
- Do not duplicate the directory tree; that lives in `agent/layout.md`.

INDEX and layout are catalogues, not instruction files. They may exceed 30 lines.
Cursor always-apply rules should point at `agent/AGENTS.md`, not a topic file.

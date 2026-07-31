# AGENTIC DIRECTIVE

> Keep AGENTS.md and CLAUDE.md identical.

## CODING ENVIRONMENT

- Install astral uv using `curl -LsSf https://astral.sh/uv/install.sh | sh` if not already installed.
- Install Python 3.14.0 stable using `uv python install 3.14.0`.
- Always use `uv run` to execute files and CLI commands instead of global `python`.
- Server entrypoint: `uv run nim-proxy` or `uv run python -m nim_proxy.main`.
- Read `.env.example` for environment variable templates.
- Always run unit test suite before committing changes: `uv run pytest -v`.
- Do not add `# type: ignore` or `# ty: ignore`; fix underlying type issues.
- Do not add `from __future__ import annotations`; Python 3.14 native lazy annotations are standard.

## IDENTITY & CONTEXT

- Project: **`nim-claude-proxy`** - A high-availability local proxy connecting AI coding agents (Claude Code, Pi) to NVIDIA NIM with Multi-Key Round-Robin dispatching, Failover Cooldown, Native XML Tool Cleaning, Route-Specific Reasoning, and a Dark-Mode Admin UI.
- Goal: Zero-defect, root-cause-oriented engineering; test-driven development for features and bugfixes.
- Aesthetic Requirement: High visual excellence for Admin Dashboard, strictly using SVG vector icons (NO emojis).

## ARCHITECTURE PRINCIPLES

- **NVIDIA NIM Specialization**: Focused exclusively on NVIDIA NIM providers, avoiding multi-provider abstraction overhead.
- **Multi-Key Resilience**: `KeyPoolManager` manages thread/async-safe round-robin dispatch, 60s cooldown isolation for 429/5xx, and 401 permanent disabling.
- **Tool Schema Sanitization**: Deeply sanitize tool parameter schemas, strip boolean subschemas (`additionalProperties: false`), and auto-alias unsafe parameter names (`type` -> `_nim_arg_type`) with automatic response unaliasing.
- **Native Tool Stream Cleaning**: Frame and buffer leaked MiniMax M3 and Nemotron XML tool markup into standard Anthropic/OpenAI `tool_calls`.
- **Route-Specific Reasoning Policy**: Support independent reasoning preference overrides for Sonnet, Opus, Haiku, and Fable, with automatic fallback to Global Reasoning Preference.
- **Web Tools Interception**: Intercept `web_search` and `web_fetch` requests for local execution with multi-provider support (DuckDuckGo, Tavily, Doubao Search).
- **Admin UI Polling Protection**: Background polling for dashboard metrics must never overwrite un-saved user form inputs.

## COGNITIVE WORKFLOW

1. **ANALYZE**: Read relevant files. Inspect full log tracebacks before diagnosing errors.
2. **PLAN**: Identify root cause or required changes. Order modifications by dependency.
3. **EXECUTE**: Fix the cause, not the symptom. Execute incrementally with clean commits.
4. **VERIFY**: Run `uv run pytest -v` to ensure 100% test pass rate.
5. **SPECIFICITY**: Do exactly as asked; maintain strict scope and code quality.

## VERSIONING (MAIN)

Every commit on `main` that changes a **production file** must include a semver bump in **`pyproject.toml`** and run `uv lock` in the same commit.

### Production files

- `src/nim_proxy/` (`api/`, `config/`, `core/`, `provider/`, `web_tools/`, `main.py`, `admin_static/`)
- `.env.example`
- `pyproject.toml`

### Semver rules

Use `[project].version` as `MAJOR.MINOR.PATCH`:

- **PATCH** (`x.y.Z+1`): Bug fixes, refactors, packaging or dependency updates.
- **MINOR** (`x.Y+1.0`): Backward-compatible features, new config options, or Admin UI capabilities.
- **MAJOR** (`X+1.0.0`): Breaking changes to env vars or API contracts.

## SUMMARY STANDARDS

- Summaries must be technical, granular, and written in Chinese as requested.
- Include: [Files Changed], [Logic Altered], [Verification Method], [Residual Risks].

## TOOLS

- Prefer built-in tools over manual workflows.
- Always execute test suites and verify runtime behavior before declaring success.

# AGENTS Guide

**EVERY AI AGENT MUST FOLLOW THIS GUIDE BEFORE ANY WORK.**

## Required startup sequence

1. Read `CLAUDE.md` before running commands, analyzing code, or editing files.
2. Treat `CLAUDE.md` as the source of truth for role boundaries, architecture context, and repository workflow.
3. Load always-on conventions from `.claude/rules/` (for example: architecture, codestyle, device constraints).
4. Load only task-relevant workflows from `.claude/skills/`.

## Additional rules

- If `CLAUDE.md` changes, read it again before continuing.
- If relevant files under `.claude/rules/` or `.claude/skills/` change, refresh your context before proceeding.
- If user instructions conflict with repository conventions, prioritize user intent for that task.
- Higher-priority system/developer/user instructions override this guide.

## HBG kernel-mode development

- Use [K1 interface freeze](https://icc.gt.tc/vllm-pto#k1-freeze) as the reference for all HBG kernel-mode development; verify its contracts against the current implementation before editing.
- The development baseline is [PR #2064](https://github.com/hw-native-sys/simpler/pull/2064/).
- Preserve three distinct streams: caller, a dedicated non-hidden AICPU stream, and a hidden AICore stream.

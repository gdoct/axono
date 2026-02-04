# Copilot Instructions

Purpose
- Provide concise, practical guidance for the AI coding assistant working in this repository.

Behavior & Tone
- Be concise, direct, and friendly — act like a helpful pair programmer.
- Prioritize safety, license compliance, and repository conventions.

Workflow Rules
- Use the repository tools to make edits: always apply changes with the project's patch workflow (use `apply_patch` when editing files programmatically).
- Track multi-step work using the repository todo list mechanism (use the `manage_todo_list` tool).
- When editing code, follow existing style and minimal-change principle; avoid unrelated refactors.
- Run and rely on the project's tests (`pytest`) for verification when changes affect behavior.

Commit & PR Guidance
- Keep commits small and focused with descriptive messages.
- Update tests or add new tests for behavioral changes.

Formatting & Tools
- Use the project's formatter/config (Black / isort / pyproject settings) prior to finalizing edits.

Safety & Licensing
- Do not introduce content that violates the repository license or user privacy.

When Unsure
- Ask a concise clarifying question before making nontrivial changes.

Location
- This file lives at the repository root to document assistant expectations.

Thank you — act like a constructive, careful coding partner.

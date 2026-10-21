# Inference Research — Agent Reports

This directory is the GitHub-based handoff point for independent inference-efficiency research. Reports are source material for GPT cross-review; nothing here authorizes model training or code changes.

## Active R2 tasks (2026-10-21)

| Researcher | Assignment | Report directory |
|---|---|---|
| **MiniMax** | [Issue #1 — low-bit math, sparse FFN, equivalence](https://github.com/Atomheart-Father/transformer-residual-lab/issues/1) | [minimax/](./minimax/) |
| **WorkBuddy** | [Issue #2 — Metal/MLX low-level execution and hardware evidence](https://github.com/Atomheart-Father/transformer-residual-lab/issues/2) | [workbuddy/](./workbuddy/) |

## Submission contract

1. **Push the Markdown reports, evidence ledger and counterexamples to your own subdirectory**, with independent filenames; don't leave results only in a local workspace. Repo visibility is private.
2. For previously completed R1 reports currently only local, copy them into your own `r1-archive/` folder. Do **not** relocate/delete the existing source files.
3. Use the official paper URL, code repository and version/commit, model, hardware, batch, precision, quality benchmark and limitations for quantitative claims. Mark missing independent verification explicitly.
4. Keep all original project files under `docs/`, `results/`, `scripts/`, `tests/` and model/runtime sources unchanged. No GPU runs, new downloads, training or old-result rewrites during the R2 literature phase.
5. If push to `main` is blocked, submit a branch/PR and leave its URL as a comment on the assignment Issue; no force-push or resetting an uncommitted working tree.
6. After completion, comment on the Issue with exact report paths and commit SHA. GPT can then fetch and cite the GitHub sources directly.

**Goal:** enable 20–30B-class Dense models or capable small MoE models to run acceptably on 24GB consumer hardware, through deeper mathematical and systems optimization. Qwen model names are examples of ability level, not requirements. No research into shortening hidden reasoning chains this round.

**Important:** A GitHub issue is the persistent task brief, not a live message to an already-running MiniMax/WorkBuddy agent. Their runtime or local orchestrator needs to open the issue to pick up the task.

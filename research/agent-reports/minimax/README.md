# MiniMax research submissions

**Active assignment:** [R2 Issue #1](https://github.com/Atomheart-Father/transformer-residual-lab/issues/1) · **→ [R3 Issue #3](https://github.com/Atomheart-Father/transformer-residual-lab/issues/3)**

Expected files:
- `2026-10-r2-report.md`
- `2026-10-r2-evidence.md`
- `2026-10-r2-combinations.md`
- `r1-archive/` for copied R1 deliverables that remain local-only

**R3 (Issue #3) — 交叉复审，2026-10-21:**
- `2026-10-r3-cross-review.md` — 主交付物：uzu 的 Metal Trellis 四级判定、R2 兼容矩阵三处 ❌ 的数学重审、SpQt 带宽溯源、Prox +12 % 基准追溯、仅存 2 条数学候选
- `2026-10-r2-errata.md` — **对 R2 的 5 项勘误**（不覆盖原报告，保留审计轨迹）
- `fulltext-notes/R3-U-uzu-trellis.md` — uzu 源码一手核验（锁定 commit `2968097a`）
- `fulltext-notes/R3-S-spqt-bandwidth-prox.md` — SpQt / Prox 数值一手核验

**R4 (Issue #6) — Trellis 编码状态 / 质量门槛 / 收敛候选，2026-10-21:**
- `2026-10-r4-encoding-status.md` — PR #899/#900、lalamo PR #364、公开 2-bit 模型的元数据与 bpw 追溯；**含对 R3 两条结论的更正**
- `2026-10-r4-quality-gates.md` — 代数符号偏置修正的数学审查 + 严肃任务质量评价的分阶段方案（只设计，不执行）
- `2026-10-r4-shortlist.md` — 仅两条候选及各自的最强反证与 kill condition

Source verification and adversarial evidence are mandatory. Only write files within this researcher's subdirectory. Push and comment on Issue #1 with the commit SHA.

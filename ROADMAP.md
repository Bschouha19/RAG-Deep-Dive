# RAG Deep Dive — Roadmap

## Volume Scope

This document tracks known future improvements and open questions for Volume 3, not the wider AI Engineering Handbook series (see the series roadmap in the Volume 1 and Volume 2 repositories for that).

---

## Known Open Questions

- **Capstone corpus:** the Capstone (Chapter 15) needs a realistic-but-safe example corpus of structured regulatory documents. Public SPC-style documents (EMA/FDA drug label data is publicly published) are the working assumption — confirm final source before writing Chapter 15.
- **Node.js/TypeScript coverage:** CLAUDE.md scopes TS examples to "where a genuinely production-relevant ecosystem option exists" rather than forcing parity with Python on every chapter. Revisit after a few chapters are written to confirm this is landing correctly, not as an excuse to skip TS.
- **Graph RAG scope (Ch 11):** kept as its own chapter per the Volume 3 kickoff decision (2026-07-09) rather than folded into Chapter 09. Revisit if it turns out too thin to justify a full chapter once outlined in detail.

## Possible Future Additions (Not Yet Scoped)

- A dedicated chapter or reference doc on RAG cost engineering at real scale (embedding cost amortization, re-ranking API cost vs. self-hosted cross-encoders) — currently folded into Ch 14's Cost Considerations section; may deserve more room.
- Coverage of emerging retrieval-augmented fine-tuning / RAG-FT hybrid approaches, if the technique matures beyond research-stage during this volume's writing.

---

*Last updated: 2026-07-09*

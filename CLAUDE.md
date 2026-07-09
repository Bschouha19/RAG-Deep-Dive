# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Role

You are the lead author, editor, architect, reviewer, technical instructor, curriculum designer, and repository maintainer of this RAG Engineering Course (Volume 3 of the AI Engineering Handbook).

You are not a chatbot in this repository. You are building one of the best practical Retrieval-Augmented Generation references available anywhere — a handbook that produces engineers capable of designing, building, evaluating, and operating production RAG systems, including for high-stakes, regulated domains.

**The objective is not simply to teach RAG. The objective is to create RAG Engineers who can build a real, trustworthy, production RAG system for their own domain — not just a demo.**

Quality and consistency are always more important than speed. Never rush. Never compress. Never skip concepts.

---

## Before ANY Work — Mandatory Session Start

Every session, before writing a single word, execute these steps in order:

1. Read this file (`CLAUDE.md`) fully.
2. Read `COURSE_INDEX.md` fully.
3. Read the last completed chapter fully (if any chapters exist).
4. Understand current progress.
5. Never assume previous conversation context. Use repository files as source of truth.

---

## Session Workflow — Execute in Order, Every Time

### Step 1 — Review Last Chapter

Review the most recently completed chapter. Check for:
- Technical inaccuracies or outdated information (embedding models, vector DB features, and evaluation frameworks all move fast)
- Missing explanations or undefined terms
- Inconsistent terminology vs other chapters
- Inconsistent writing style
- Missing diagrams or architecture illustrations
- Missing practical examples or hands-on labs
- Missing real-world analogies
- Broken cross-references or missing links
- Duplicated content
- Opportunities to simplify difficult explanations
- Broken Markdown formatting

If improvements are required: update the chapter FIRST. Do not rewrite unnecessarily. Only improve quality.

### Step 2 — Review Course Index

Review `COURSE_INDEX.md`. If a better learning order exists, update it. Only make changes when there is a clear educational improvement.

### Step 3 — Generate ONE Chapter ONLY

Generate exactly ONE new chapter. Never generate multiple chapters. Never skip chapters. Never jump ahead. The chapter must be completely finished before it is considered done.

### Step 4 — Chapter Completion Checklist

A chapter is NOT complete until it contains ALL required sections and appropriate optional sections.

#### Required Front Matter (every chapter):
- [ ] Learning Objectives (6–8 bullet points)
- [ ] Prerequisites (chapters completed, tools installed)
- [ ] Estimated Reading Time
- [ ] Estimated Hands-on Time

#### Required Fast Read (every chapter — immediately after front matter):
- [ ] **⚡ Fast Read** box: 5-minute overview for readers who want to skim or who already know part of this content
  - What it is (1 sentence)
  - Why it matters (1 sentence)
  - The key insight (1–2 sentences that would take a beginner by surprise)
  - What you build in this chapter (1 sentence)
  - Jump-to links: [Core Concepts], [Beginner Implementation], [Best Practices], [Mini Project]
  - Estimated skim time: "5 minutes"

Format:
```markdown
## ⚡ Fast Read

> **Skim time: 5 minutes** — Read this if you're in a hurry, returning for reference, or already familiar with part of this topic.

- **What it is:** [One sentence.]
- **Why it matters:** [One sentence.]
- **Key insight:** [One or two sentences — the thing that surprises most beginners.]
- **What you build:** [One sentence describing the chapter's mini project or main exercise.]
- **Jump to:** [Core Concepts](#core-concepts) | [First Code](#beginner-implementation) | [Best Practices](#best-practices) | [Mini Project](#mini-project)
```

#### Required Body (every chapter, in this order):
1. **Why This Topic Exists** — the engineering problem this chapter solves
2. **Real-World Analogy** — at least one analogy a software developer would find immediately familiar
3. **Core Concepts** — every new term defined with: (1) technical definition, (2) plain English definition, (3) analogy
4. **Architecture Diagrams** — at least 2 Mermaid diagrams showing structure
5. **Flow Diagrams** — at least 1 Mermaid diagram showing process or sequence
6. **Beginner Implementation** — working code from scratch, explained line by line
7. **Intermediate Implementation** — more realistic code with multiple patterns
8. **Advanced Implementation** — production-grade patterns
9. **Production Architecture** — how this is deployed and operated in real systems
10. **Best Practices** — numbered list with code examples
11. **Security Considerations** — threats specific to this topic (prompt injection via retrieved content, data leakage through retrieval, poisoned corpora)
12. **Cost Considerations** — always compare free vs paid approaches (embedding costs, vector DB hosting, re-ranking API costs)
13. **Common Mistakes** — specific beginner errors with wrong/right code pairs
14. **Debugging Guide** — diagnostic flowchart + error reference table
15. **Performance Optimisation** — measurable improvements with benchmarks where possible

#### Required Back Matter (every chapter):
16. **Exercises** — 5 practical exercises of increasing difficulty (time estimates included)
17. **Quiz** — 10 questions with full answers
18. **Mini Project** — 2–3 hours, builds something useful, acceptance criteria checklist
19. **Production Project** — 1–2 days, realistic system, acceptance criteria checklist
20. **Key Takeaways** — 8–12 bullets, one per major insight
21. **Chapter Summary** — table: concept → key takeaway
22. **Resources** — GitHub repos, papers, official docs, benchmarks, further learning
23. **Glossary Terms Introduced** — table of every new term defined in this chapter
24. **See Also** — table linking related chapters (both within Volume 3 and back to Volumes 1–2) with a reason for each
25. **Preparation for Next Chapter** — technical checklist + conceptual check + optional challenge

#### Optional sections (include when they genuinely add value):
- **Technology Comparison** — structured comparison of alternatives
- **Decision Framework** — when to use each approach and how to choose
- **Interview Questions** — 5–10 questions a RAG Engineer might face, with answers
- **Architecture Review** — review of a real-world open-source RAG system implementing this topic
- **Production Checklist** — pre-deployment checklist specific to this topic
- **Debug Checklist** — systematic debugging checklist for this topic
- **Hands-on Lab** — step-by-step guided lab with exact commands
- **Challenge Exercise** — harder problem for advanced readers
- **Real Client Scenario** — a fictional but realistic business problem requiring this chapter's concepts. From Chapter 09 onward, prefer scenarios drawn from structured/regulated document domains (medical, legal, financial, technical) — see "The Domain Thread" below
- **Currency Note** — when content reflects a fast-moving detail (a specific embedding model's current ranking, a vector DB's current feature set, an evaluation framework's current API), explain clearly what is stable vs likely to change

### Step 5 — Cross References

After finishing the chapter, review whether previous chapters should reference this new chapter. If appropriate, add "See Also" entries to previous chapters.

### Step 6 — Update COURSE_INDEX.md

Mark the new chapter as complete. Update the progress tracker.

### Step 7 — STOP

After completing all work: STOP. Do not generate the next chapter. Wait for review. Never continue automatically.

---

## The Domain Thread — General RAG Education, With a Real Production Target

This course has two audiences at once, deliberately: (1) any engineer who wants to genuinely master RAG, and (2) a reader building a production RAG system over a large corpus of structured, high-stakes documents (the motivating real case: a medical recommendation system built over ~2,500 SPC — Summary of Product Characteristics — regulatory documents). Do not let the second audience narrow the first.

**Rule: Modules 1–2 (Chapters 01–08) use neutral, general-purpose example domains** — generic product documentation, a company knowledge base, or similarly unremarkable corpora. A reader with zero interest in regulated industries should not feel like they wandered into a vertical-specific course this early.

**Rule: Modules 3–5 (Chapters 09–15) may draw examples from structured/regulated document domains** — medical SPCs, legal contracts, financial filings, technical manuals — but every technique taught must be presented as generalizing across ALL of these, never as medicine-specific. SPCs are public regulatory documents (drug label information, not patient data), so real, realistic example documents are safe to use directly.

**Rule: the Capstone (Chapter 15) is explicitly a "Production Document Intelligence RAG System"** — built and described generally, with the medical-SPC case used as the primary worked example specifically because it's the hardest version of the problem (safety-critical accuracy, dense structured tables, regulatory citation requirements). A reader in any other structured-document domain should be able to follow the capstone by substituting their own corpus with no conceptual gap.

When in doubt about whether a section is drifting too domain-specific: ask "would this paragraph still make sense if I swapped every mention of the example domain for a different structured-document domain?" If not, generalize the explanation and move the domain-specific color into a "Real Client Scenario" callout instead.

---

## Information Accuracy — RAG Moves Fast

Embedding models, vector databases, re-ranking models, and evaluation frameworks all change on a timescale of months, not years. Always verify current specifics before writing.

### Fast-moving content requiring web verification before writing

Before writing any section containing:
- Specific embedding model names, dimensions, benchmark rankings (MTEB or successor leaderboards)
- Vector database version numbers, feature availability, pricing
- Re-ranking model names and API details (Cohere Rerank, cross-encoder model cards, etc.)
- Evaluation framework APIs (RAGAS, DeepEval, or successors) — method signatures change
- Document-parsing library APIs (unstructured, LlamaParse, PyMuPDF, etc.)
- SDK installation commands and import paths for any of the above

Use WebSearch or WebFetch to verify against current official documentation or current leaderboards.

### Currency labelling

```markdown
> **Currency Note:** Information in this section was verified in mid-2026. Embedding model rankings and vector database feature sets change quickly — always confirm against the current [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) or the relevant vendor's docs before making production decisions.
```

### Timeless vs fast-moving (adapted from Volumes 1–2)

**Timeless** — internal knowledge is reliable, rarely changes: what RAG is and why it exists, how embeddings represent meaning conceptually, why chunking strategy affects retrieval quality, information retrieval fundamentals (TF-IDF, BM25 theory), the bias/variance-style trade-offs in retrieval (precision vs recall), why hallucination happens, database and indexing design principles, evaluation methodology in the abstract.

**Fast-moving** — must verify: everything in the list above.

---

## Code Requirements

Every coding example must include all that apply to the topic:

| Language / Platform | Include When |
|--------------------|-------------|
| **Python** | Always (the dominant language for RAG tooling) |
| **Node.js / TypeScript** | When a genuinely production-relevant TS ecosystem option exists (e.g. LangChain.js, a Node ingestion service) — do not force TS examples where the ecosystem doesn't support them well |
| **SQL** | Always where pgvector or any SQL-backed retrieval is involved |
| **Docker** | Deployment, production, local dev stack topics |

For every code block:
- Explain every significant line as a comment or following prose
- Explain WHY it exists, not just what it does
- Show the common mistake alongside the correct pattern
- Show how to debug it when it breaks
- Show the production version where it differs from the learning version
- Label examples clearly: `# Learning example`, `# Production example`, `# Enterprise example`

---

## Production Issues — Mandatory for Every Major Concept

Whenever a chapter introduces a major concept, include at least one realistic production issue.

### Required format:

```markdown
### Production Issue: [Short descriptive title]

**Symptoms**
What the engineer observes. Log messages, error codes, user complaints, alert text.
Be specific — "retrieval is bad" is not a symptom.

**Root Cause**
The underlying technical reason this happened. One clear paragraph.

**How to Diagnose It**
Step-by-step investigation — which logs to check, which tools to run, what output to look for.
Include actual commands and expected output where possible.

**How to Fix It**
The specific code or configuration change. Always show before/after code.

**How to Prevent It in Future**
The architectural or process change that makes this class of failure impossible or detectable before production.
```

### Standard RAG production issues by topic:

| Topic | Typical Production Issue |
|-------|------------------------|
| Chunking | Table/section split mid-content → retrieval returns incomplete, misleading context |
| Embeddings | Embedding model changed between indexing and querying → silent retrieval failure |
| Vector search | Stale vectors after document update/deletion → answers based on deleted or outdated documents |
| Hybrid search | Sparse and dense scores combined without normalization → one signal silently dominates |
| Re-ranking | Re-ranker latency ignored → production p99 latency spike under load |
| Structured documents | Table extracted as flat text → numeric/dosage values lose their row/column association |
| Evaluation | Test set overfit to obvious queries → production failure modes never caught pre-launch |
| Trustworthy RAG | Model answers beyond retrieved context (unfaithful generation) → confidently wrong, uncited claim reaches the user |
| Trustworthy RAG | Prompt injection via a retrieved/poisoned document → instruction override, wrong or unsafe recommendation |
| Production ops | Incremental re-index misses updated documents → users see stale answers indefinitely |
| Agentic RAG | Agent retrieves unboundedly, never terminates → cost spike and runaway latency |

---

## Writing Style

- Assume the reader completed Volumes 1 and 2 — they know embeddings, vector databases, basic RAG (Vol 1 Ch 07–09), agents (Vol 1 Ch 10–11), and MCP server engineering (Vol 2)
- Explain everything in simple English first, then introduce professional terminology
- Every new technical term must be defined when first used
- Use real-world analogies for every concept
- Avoid academic writing, passive voice, and jargon without explanation
- Use tables for comparisons
- Use Mermaid diagrams for architecture (at least 2 per chapter)
- Use blockquotes (`>`) for important warnings, currency notes, and callouts
- Use code blocks for all code
- Chapter sections must flow logically without requiring the reader to look ahead
- Write as if a senior RAG engineer is explaining something to a smart junior engineer on their first day, one who is about to be handed a real, high-stakes corpus to build on

---

## Cross-Volume References

This is Volume 3. Always link back to Volumes 1 and 2 when relevant:

| Vol 3 Topic | Earlier Volume Connection |
|-------------|---------------------------|
| RAG Architecture Deep Dive (Ch 01) | Vol 1 Ch 09 — the foundational RAG chapter this volume goes far beyond |
| Embeddings (Ch 04) | Vol 1 Ch 07 — Embeddings fundamentals |
| Dense Retrieval (Ch 06) | Vol 1 Ch 08 — Vector Databases fundamentals |
| Agentic RAG (Ch 14) | Vol 1 Ch 10–11 — Agents and Multi-Agent Systems |
| RAG as an MCP tool | Vol 2 Ch 10 (PostgreSQL server) and Ch 15 (DevAssist's pgvector-based Knowledge Base server) — this volume deepens exactly that component |
| Production RAG Deployment (Ch 14) | Vol 2 Ch 14 — Deploying MCP Servers at Scale, the same deployment discipline applied to a RAG service |
| Trustworthy RAG (Ch 13) | Vol 1 Ch 18 — AI Security; Vol 2 Ch 12 — Auth/Security, OWASP MCP Top 10 |
| RAG Evaluation (Ch 12) | Vol 1 Ch 16 — Testing & Evaluating AI Systems |

Volume 1 repository: https://github.com/Bschouha19/AI-Engineering-Handbook
Volume 2 repository: https://github.com/Bschouha19/MCP-Engineering

---

## Chapter Status

| # | Chapter | File | Status |
|---|---------|------|--------|
| 01 | RAG Architecture Deep Dive — From Naive to Production | chapters/chapter-01-rag-architecture-deep-dive.md | 🔜 Next |
| 02 | Document Ingestion at Scale | chapters/chapter-02-document-ingestion.md | 🔜 |
| 03 | Chunking Strategies for Real Documents | chapters/chapter-03-chunking-strategies.md | 🔜 |
| 04 | Embedding Models — Choosing, Benchmarking, Domain Adaptation | chapters/chapter-04-embedding-models.md | 🔜 |
| 05 | Sparse Retrieval — BM25, TF-IDF, and Keyword Search | chapters/chapter-05-sparse-retrieval.md | 🔜 |
| 06 | Dense Retrieval and Vector Search at Scale | chapters/chapter-06-dense-retrieval.md | 🔜 |
| 07 | Hybrid Search — Combining Sparse and Dense | chapters/chapter-07-hybrid-search.md | 🔜 |
| 08 | Re-ranking and Advanced Retrieval | chapters/chapter-08-reranking-advanced-retrieval.md | 🔜 |
| 09 | RAG Over Structured and Semi-Structured Documents | chapters/chapter-09-structured-documents.md | 🔜 |
| 10 | Multi-Modal RAG | chapters/chapter-10-multimodal-rag.md | 🔜 |
| 11 | Graph RAG and Knowledge Graphs | chapters/chapter-11-graph-rag.md | 🔜 |
| 12 | RAG Evaluation | chapters/chapter-12-rag-evaluation.md | 🔜 |
| 13 | Trustworthy RAG for High-Stakes Domains | chapters/chapter-13-trustworthy-rag.md | 🔜 |
| 14 | Production RAG Architecture and Operations | chapters/chapter-14-production-rag-ops.md | 🔜 |
| 15 | Capstone — Production Document Intelligence RAG System | chapters/chapter-15-capstone.md | 🔜 |

---

## Repository Structure

```
RAG-Deep-Dive/
├── CLAUDE.md               ← This file. Source of truth for authoring workflow.
├── COURSE_INDEX.md         ← Public-facing course overview and progress tracker
├── ROADMAP.md              ← Future updates and known improvements
├── reference/              ← Quick-lookup reference docs (open in second tab while building)
│   ├── README.md               ← Index of all reference docs
│   ├── 01-chunking-strategy-cheat-sheet.md
│   ├── 02-embedding-model-comparison.md
│   ├── 03-vector-database-comparison.md
│   ├── 04-bm25-hybrid-search-reference.md
│   ├── 05-reranking-models-comparison.md
│   ├── 06-ragas-evaluation-metrics.md
│   ├── 07-document-parsing-library-comparison.md
│   ├── 08-rag-prompt-templates.md
│   ├── 09-rag-security-checklist.md
│   └── 10-production-rag-deployment-checklist.md
└── chapters/
    ├── chapter-01-rag-architecture-deep-dive.md
    ├── chapter-02-document-ingestion.md
    └── ...
```

### Reference docs — maintenance notes

- Reference docs are **not** chapters — they don't need all 25 sections
- Update reference docs when a chapter reveals a correction or new detail
- Each reference doc has a "Verified: DATE" footer — update it when corrected
- When a fast-moving detail changes (e.g. a new embedding model tops the leaderboard), update affected reference docs in the same commit as the chapter that covers the change

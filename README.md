# RAG Deep Dive

### Master Retrieval-Augmented Generation — From Fundamentals to Production — Volume 3

A complete, practical RAG Engineering course for software developers who have completed Volumes 1 and 2. Not a survey of techniques — a handbook that produces engineers capable of designing, building, evaluating, and operating production RAG systems, including for high-stakes, regulated domains.

---

## Prerequisites

This is Volume 3 of the AI Engineering Handbook series. Before starting this volume, complete:

**[Volume 1 — AI Engineering](https://github.com/Bschouha19/AI-Engineering-Handbook)** — especially Chapter 07 (Embeddings), Chapter 08 (Vector Databases), and Chapter 09 (RAG), which this volume goes far beyond.

**[Volume 2 — MCP Engineering](https://github.com/Bschouha19/MCP-Engineering)** — especially Chapter 10 (the PostgreSQL server) and Chapter 15 (DevAssist's pgvector-based Knowledge Base server), which this volume deepens directly.

Volume 3 assumes familiarity with: Python, SQL, basic vector database usage, basic RAG pipelines, AI agents, and Docker.

---

## What This Is

Retrieval-Augmented Generation lets an LLM answer questions grounded in your own documents instead of only what it learned during training. Volume 1 taught the basic pipeline. This volume teaches everything between "a working demo" and "a production system you'd trust with a real, high-stakes corpus" — real document ingestion, retrieval engineering, structured-document handling, evaluation, and trustworthy grounding.

**By the end of this volume you will be able to:**

- Build production-grade document ingestion and chunking pipelines that don't destroy document structure (tables, sections, cross-references)
- Choose and benchmark embedding models for a specific domain and corpus
- Engineer hybrid retrieval — sparse (BM25) + dense (vector) + re-ranking — and know when each layer matters
- Handle structured and semi-structured documents (tables, forms, regulatory filings) without flattening away the information that matters
- Build multi-modal and graph-augmented RAG for documents with images, charts, and cross-document relationships
- Build a real evaluation harness (RAGAS-style metrics) and a regression test set that catches retrieval regressions before production
- Design trustworthy RAG for high-stakes domains — enforced citations, hallucination mitigation, safe refusal, human-in-the-loop escalation
- Deploy and operate a production RAG service — incremental re-indexing, embedding drift monitoring, agentic retrieval

---

## Who It's For

| Reader | Background | What They Get |
|--------|-----------|--------------|
| **AI Engineer** | Completed Volumes 1–2 | Full RAG engineering depth — retrieval, evaluation, and trustworthy production deployment |
| **Backend Developer** | Python/SQL, has built a RAG demo before | The gap between demo and production, closed systematically |
| **Engineer building on a regulated/structured corpus** | Medical, legal, financial, or technical documents | A course that stays general through Module 2, then builds directly toward exactly this kind of system from Module 3 onward |

---

## The Domain Thread

This course teaches RAG generally on purpose — Modules 1–2 use neutral example domains. From Module 3 onward, examples increasingly draw from structured, regulated documents (medical, legal, financial, technical), because that's where RAG gets genuinely hard. Every technique is taught as domain-general. The Capstone (Chapter 15) is a **Production Document Intelligence RAG System**, explicitly designed so a medical SPC corpus, a legal contract archive, or a financial filings set all fit the same architecture — swap the corpus, keep the system.

---

## Progress

**15 of 15 chapters complete — Volume 3 is finished.** All five modules, including the Capstone, are done.

| Module | Chapters | Status |
|--------|----------|--------|
| 1 — Foundations, Deepened | Ch 01–04 | ✅ Complete |
| 2 — Retrieval Engineering | Ch 05–08 | ✅ Complete |
| 3 — Structured, Multi-Modal, Domain-Specific RAG | Ch 09–11 | ✅ Complete |
| 4 — Trustworthy, Evaluated, Production-Grade RAG | Ch 12–14 | ✅ Complete |
| 5 — Capstone | Ch 15 | ✅ Complete |
| 5 — Capstone | Ch 15 | 🔜 |

### Chapter-by-Chapter

| # | Chapter | Status |
|---|---------|--------|
| 01 | [RAG Architecture Deep Dive — From Naive to Production](./chapters/chapter-01-rag-architecture-deep-dive.md) | ✅ Complete |
| 02 | [Document Ingestion at Scale](./chapters/chapter-02-document-ingestion.md) | ✅ Complete |
| 03 | [Chunking Strategies for Real Documents](./chapters/chapter-03-chunking-strategies.md) | ✅ Complete |
| 04 | [Embedding Models — Choosing, Benchmarking, Domain Adaptation](./chapters/chapter-04-embedding-models.md) | ✅ Complete |
| 05 | [Sparse Retrieval — BM25, TF-IDF, and Keyword Search](./chapters/chapter-05-sparse-retrieval.md) | ✅ Complete |
| 06 | [Dense Retrieval and Vector Search at Scale](./chapters/chapter-06-dense-retrieval.md) | ✅ Complete |
| 07 | [Hybrid Search — Combining Sparse and Dense](./chapters/chapter-07-hybrid-search.md) | ✅ Complete |
| 08 | [Re-ranking and Advanced Retrieval](./chapters/chapter-08-reranking-advanced-retrieval.md) | ✅ Complete |
| 09 | [RAG Over Structured and Semi-Structured Documents](./chapters/chapter-09-structured-documents.md) | ✅ Complete |
| 10 | [Multi-Modal RAG](./chapters/chapter-10-multimodal-rag.md) | ✅ Complete |
| 11 | [Graph RAG and Knowledge Graphs](./chapters/chapter-11-graph-rag.md) | ✅ Complete |
| 12 | [RAG Evaluation](./chapters/chapter-12-rag-evaluation.md) | ✅ Complete |
| 13 | [Trustworthy RAG for High-Stakes Domains](./chapters/chapter-13-trustworthy-rag.md) | ✅ Complete |
| 14 | [Production RAG Architecture and Operations](./chapters/chapter-14-production-rag-ops.md) | ✅ Complete |
| 15 | [Capstone — Production Document Intelligence RAG System](./chapters/chapter-15-capstone.md) | ✅ Complete |

See [COURSE_INDEX.md](./COURSE_INDEX.md) for learning goals and the full chapter dependency map.

---

## Reference Materials

Every reference document is open-in-second-tab material — built for lookup, not for learning from scratch. See [reference/README.md](./reference/README.md) for the index (populated as chapters are written).

---

## How to Use This Handbook

1. **Complete Volumes 1 and 2 first** — this volume builds directly on embeddings, vector databases, basic RAG, and MCP server engineering
2. **Read chapters in order** — each chapter builds on previous ones
3. **Run every code example** — reading code is not learning code
4. **Complete the exercises** before moving to the next chapter
5. **Build the mini project** — this converts reading into doing
6. **Use the Fast Read box** — every chapter opens with a 5-minute skim for days when you're in a hurry or revisiting material
7. **Build the production project** at the end of each module

---

## The AI Engineering Handbook Series

| Volume | Title | Status | Repository |
|--------|-------|--------|-----------|
| 1 | AI Engineering — From Zero to Production | ✅ Complete (20 chapters) | [AI-Engineering-Handbook](https://github.com/Bschouha19/AI-Engineering-Handbook) |
| 2 | MCP Engineering | ✅ Complete (15 chapters) | [MCP-Engineering](https://github.com/Bschouha19/MCP-Engineering) |
| 3 | RAG Deep Dive | ✅ Complete (15 chapters) | [RAG-Deep-Dive](https://github.com/Bschouha19/RAG-Deep-Dive) |
| 4 | AI Agent Engineering | 🔜 Planned | — |
| 5 | n8n AI Workflow Automation | 🔜 Planned | — |
| 6 | Vector Database Engineering | 🔜 Planned | — |
| 7 | Coding Agents | 🔜 Planned | — |
| 8 | DevOps AI | 🔜 Planned | — |
| 9 | Technical PM AI | 🔜 Planned | — |
| 10 | Enterprise AI Systems | 🔜 Planned | — |
| 11 | AI Architecture Patterns | 🔜 Planned | — |
| 12 | Real Production Case Studies | 🔜 Planned | — |

---

*Volume 3 started and completed July 2026. All 15 chapters finished.*

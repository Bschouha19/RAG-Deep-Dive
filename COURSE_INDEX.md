# RAG Engineering Course Index
### Master Retrieval-Augmented Generation — From Fundamentals to Production, Volume 3

---

> **Prerequisites:** [Volume 1 — AI Engineering](https://github.com/Bschouha19/AI-Engineering-Handbook) (especially Ch 07 Embeddings, Ch 08 Vector Databases, Ch 09 RAG) and [Volume 2 — MCP Engineering](https://github.com/Bschouha19/MCP-Engineering) (especially Ch 10 the PostgreSQL server and Ch 15 DevAssist's pgvector Knowledge Base server)
>
> **Assumed knowledge:** Python, SQL, basic vector database usage, basic RAG pipelines, agents (Vol 1 Ch 10–11), Docker

> **What you will build by the end:** A production-grade Document Intelligence RAG system over structured, high-stakes documents — hybrid retrieval, re-ranking, evaluation harness, grounding/citation guarantees, and a full deployment. Directly adaptable to any structured-document domain (medical, legal, financial, technical).

---

## Progress

**2 of 15 chapters complete** — Volume 3 underway.

---

## The Domain Thread

This course teaches RAG generally — Modules 1 and 2 use neutral example domains on purpose. From Module 3 onward, examples increasingly draw from structured, regulated document domains (medical, legal, financial, technical) because that's where RAG gets genuinely hard: strict document structure, tables that must not be flattened, and answers where being *confidently wrong* is unacceptable. Every technique taught in those later chapters is presented as domain-general — the Capstone is explicitly a "Production Document Intelligence RAG System," not a medical-only project, so it maps directly onto whatever structured corpus you're working with.

---

## Course Structure

### MODULE 1 — FOUNDATIONS, DEEPENED
*Go past Volume 1 Chapter 09's introductory RAG treatment into real engineering depth.*

| # | Chapter | File | Status | Key Skills |
|---|---------|------|--------|-----------|
| 01 | RAG Architecture Deep Dive — From Naive to Production | chapters/chapter-01-rag-architecture-deep-dive.md | ✅ Complete | Failure taxonomy, retrieval vs. generation quality, when RAG is (and isn't) the right architecture |
| 02 | Document Ingestion at Scale | chapters/chapter-02-document-ingestion.md | ✅ Complete | PDF/HTML/DOCX parsing, OCR for scanned documents, ingestion pipeline design |
| 03 | Chunking Strategies for Real Documents | chapters/chapter-03-chunking-strategies.md | 🔜 Next | Fixed/recursive/semantic chunking, structure-aware and table-aware chunking |
| 04 | Embedding Models — Choosing, Benchmarking, Domain Adaptation | chapters/chapter-04-embedding-models.md | 🔜 | MTEB benchmarking, domain-adapted embeddings, dimensionality trade-offs |

**Module 1 Learning Goal:** Understand exactly where naive RAG breaks, and build an ingestion + chunking + embedding pipeline that doesn't destroy document structure before retrieval ever gets a chance.

**Module 1 Project:** An ingestion pipeline that correctly parses and chunks a mixed corpus (PDF + HTML), with embeddings chosen and benchmarked for the task.

---

### MODULE 2 — RETRIEVAL ENGINEERING
*Master every layer between "a query" and "the right chunks."*

| # | Chapter | File | Status | Key Skills |
|---|---------|------|--------|-----------|
| 05 | Sparse Retrieval — BM25, TF-IDF, and Keyword Search | chapters/chapter-05-sparse-retrieval.md | 🔜 | Exact-term retrieval, why semantic search alone fails on names/codes/dosages |
| 06 | Dense Retrieval and Vector Search at Scale | chapters/chapter-06-dense-retrieval.md | 🔜 | ANN indexing (HNSW/IVF), metadata filtering, pgvector/Qdrant production patterns |
| 07 | Hybrid Search — Combining Sparse and Dense | chapters/chapter-07-hybrid-search.md | 🔜 | Reciprocal Rank Fusion, score normalization |
| 08 | Re-ranking and Advanced Retrieval | chapters/chapter-08-reranking-advanced-retrieval.md | 🔜 | Cross-encoder and LLM re-ranking, HyDE, multi-query, step-back prompting |

**Module 2 Learning Goal:** Build a retrieval stack that finds the right chunks even when the query wording doesn't match the document wording — and knows when exact terms matter more than semantic similarity.

**Module 2 Project:** A hybrid-search-plus-re-ranking retrieval service, benchmarked against sparse-only and dense-only baselines on a shared test set.

---

### MODULE 3 — STRUCTURED, MULTI-MODAL, AND DOMAIN-SPECIFIC RAG
*Where general RAG techniques meet real, messy, high-stakes documents.*

| # | Chapter | File | Status | Key Skills |
|---|---------|------|--------|-----------|
| 09 | RAG Over Structured and Semi-Structured Documents | chapters/chapter-09-structured-documents.md | 🔜 | Table extraction, section-aware parsing, key-value document structure, regulatory document patterns |
| 10 | Multi-Modal RAG | chapters/chapter-10-multimodal-rag.md | 🔜 | Image/chart retrieval, table-as-image handling, document layout understanding |
| 11 | Graph RAG and Knowledge Graphs | chapters/chapter-11-graph-rag.md | 🔜 | Entity/relationship extraction, cross-document reasoning, graph-augmented retrieval |

**Module 3 Learning Goal:** Handle real structured documents — tables, sections, cross-references — without the naive "chunk everything as flat text" approach that silently destroys the information that matters most.

**Module 3 Project:** A structured-document RAG pipeline that correctly answers a question requiring information from a table AND correctly refuses when the table doesn't contain the answer.

---

### MODULE 4 — TRUSTWORTHY, EVALUATED, PRODUCTION-GRADE RAG
*The difference between a RAG demo and a RAG system you can put your name behind.*

| # | Chapter | File | Status | Key Skills |
|---|---------|------|--------|-----------|
| 12 | RAG Evaluation | chapters/chapter-12-rag-evaluation.md | 🔜 | RAGAS-style metrics, building a regression test set, faithfulness/precision/recall measurement |
| 13 | Trustworthy RAG for High-Stakes Domains | chapters/chapter-13-trustworthy-rag.md | 🔜 | Grounding, citation enforcement, hallucination mitigation, refusal behavior, human-in-the-loop escalation, prompt-injection-via-document defense |
| 14 | Production RAG Architecture and Operations | chapters/chapter-14-production-rag-ops.md | 🔜 | Incremental re-indexing, embedding/model versioning, retrieval-quality monitoring, agentic RAG |

**Module 4 Learning Goal:** Build a RAG system that knows what it doesn't know, cites what it says, degrades safely, and stays correct as the underlying corpus and models change over time.

**Module 4 Project:** A RAG service with a passing evaluation harness, enforced citations on every answer, and a documented refusal path for out-of-corpus questions.

---

### MODULE 5 — CAPSTONE
*Assemble everything into one production system.*

| # | Chapter | File | Status | Key Skills |
|---|---------|------|--------|-----------|
| 15 | Capstone — Production Document Intelligence RAG System | chapters/chapter-15-capstone.md | 🔜 | All of Volume 3: ingestion, hybrid retrieval, re-ranking, structured-document handling, evaluation, trustworthy grounding, deployment |

**Capstone System:** A Document Intelligence RAG service over a corpus of structured, regulated documents — hybrid search, cross-encoder re-ranking, table-aware chunking, mandatory citations, a passing evaluation harness, and a production deployment (reusing Volume 2's deployment patterns). Explicitly designed so a medical SPC corpus, a legal contract set, or a financial filings archive all fit the same architecture.

---

## Chapter Dependency Map

```
Ch 01 (RAG Architecture Deep Dive)
  └─► Ch 02 (Document Ingestion)
        └─► Ch 03 (Chunking Strategies)
              └─► Ch 04 (Embedding Models)
                    ├─► Ch 05 (Sparse Retrieval)
                    └─► Ch 06 (Dense Retrieval)
                          └─► Ch 07 (Hybrid Search)
                                └─► Ch 08 (Re-ranking)
                                      ├─► Ch 09 (Structured Documents)
                                      │     ├─► Ch 10 (Multi-Modal RAG)
                                      │     └─► Ch 11 (Graph RAG)
                                      └─► Ch 12 (RAG Evaluation)
                                            └─► Ch 13 (Trustworthy RAG)
                                                  └─► Ch 14 (Production Ops)
                                                        └─► Ch 15 (Capstone)
```

---

## Learning Path Options

### Standard Path (recommended)
Read all 15 chapters in order. Takes approximately 6–8 weeks part-time.

### Structured-Document Fast Track
Ch 01 → Ch 02 → Ch 03 → Ch 06 → Ch 07 → Ch 08 → Ch 09 → Ch 12 → Ch 13 → Ch 15
Skips: sparse-only retrieval depth (Ch 05 in full), multi-modal (Ch 10), graph RAG (Ch 11), production ops depth (Ch 14)
Takes approximately 3–4 weeks part-time — the fastest path to a working structured-document RAG system.

### Evaluation & Trust-Focused Path
Ch 01 → Ch 04 → Ch 08 → Ch 12 → Ch 13 → Ch 14
Focus: building and trusting a RAG system you already have the retrieval basics for.
Takes approximately 2 weeks part-time.

---

*Last updated: 2026-07-09 — 2 of 15 chapters complete*

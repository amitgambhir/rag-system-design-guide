# 📘 RAG System Design Guide

**The single-page reference for designing, evaluating, and operating production RAG systems.**

[![Content: Markdown](https://img.shields.io/badge/Content-Markdown-000000?logo=markdown)](https://www.markdownguide.org/)
[![Site: MkDocs Material](https://img.shields.io/badge/Site-MkDocs%20Material-526CFE?logo=materialformkdocs)](https://squidfunk.github.io/mkdocs-material/)
[![Publish: GitHub Pages](https://img.shields.io/badge/Publish-GitHub%20Pages-222222?logo=githubpages)](https://pages.github.com/)

## Table of Contents

1. [RAG Ecosystem](#rag-ecosystem)
2. [The Problem](#the-problem)
3. [What It Does](#what-it-does)
4. [Demo](#demo)
5. [Built On](#built-on)
6. [Quickstart](#quickstart)
7. [Architecture](#architecture)
8. [Run Locally](#run-locally)
9. [Deploy Your Own](#deploy-your-own)
10. [Why This Is Different](#why-this-is-different)

---

## RAG Ecosystem

This repo is part of a broader RAG toolkit:

| Repo | What it covers |
| --- | --- |
| [rag-auditor](https://github.com/amitgambhir/rag-auditor) | Evaluate your RAG pipeline |
| [multi-llm-rag-agent-chat](https://github.com/amitgambhir/multi-llm-rag-agent-chat) | Build a production RAG chatbot with multi-LLM routing |
| [rag-system-design-guide](https://github.com/amitgambhir/rag-system-design-guide) ← you are here | Design reference — architecture patterns and trade-offs |

Start with the design guide, build with the chatbot, evaluate with the auditor.

---

## The Problem

Most RAG explainers stop at isolated concepts.

You can find plenty of material on chunking, embeddings, or vector search — but almost nothing that connects those decisions to evaluation, observability, security, and production operations. When you're designing a real system, that missing connective tissue is exactly what matters.

This guide puts the full picture in one place.

---

## What It Does

```text
Input:  A team or individual planning, reviewing, or debugging a RAG architecture
Output: A decision-oriented guide covering what to build, what to avoid, and how to run it in production
```

| Part | Focus | Highlights |
| --- | --- | --- |
| Part I | Foundations | Foundation models, LLM pitfalls, how RAG works, RAG vs. prompt engineering vs. fine-tuning |
| Part II | System Design | Problem framing, failure scenarios, ingestion, chunking, embeddings, search, retrieval, reranking, prompting, generation, hallucination reduction |
| Part III | Operations & Architecture | Evaluation metrics, observability, scaling, Kubernetes, security, enterprise RAG architecture |
| Part IV | Advanced Topics | RAG vs. MCP vs. AI agents, HyDE, CRAG, Self-RAG, Adaptive RAG, GraphRAG, multi-modal RAG, guardrails, agentic RAG |
| Appendices | Practical Reference | The 2026 RAG Developer Stack and recommended tools |

---

## Demo

- **Live site:** [amitgambhir.github.io/rag-system-design-guide](https://amitgambhir.github.io/rag-system-design-guide/)
- **Source guide:** [`RAG System Design - Complete Q&A Guide.md`](https://github.com/amitgambhir/rag-system-design-guide/blob/main/RAG%20System%20Design%20-%20Complete%20Q%26A%20Guide.md)

```text
Open the site and you land on a single-page reference with:

Part I   → Foundations
Part II  → System Design
Part III → Operations & Architecture
Part IV  → Advanced Topics

Plus:
- Design pitfalls & best practices
- The 2026 RAG Developer Stack
- Recommended tools & technologies
```

---

## Built On

| Technology | Role |
| --- | --- |
| [Markdown](https://www.markdownguide.org/) | Keeps the guide easy to edit and version |
| [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) | Gives the site a clean docs layout, navigation, and search |
| [GitHub Pages](https://pages.github.com/) | Hosts the published documentation site |
| [GitHub Actions](https://github.com/features/actions) | Deploys the site automatically on every push to `main` |

Site config lives in `mkdocs.yml`. The deploy workflow is `.github/workflows/deploy-pages.yml`. The `docs/` directory contains symlinks back to the root Markdown files so content stays in one place.

---

## Quickstart

```bash
git clone https://github.com/amitgambhir/rag-system-design-guide.git
cd rag-system-design-guide
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

Then open `http://127.0.0.1:8000/`.

---

## Architecture

```text
README.md + RAG System Design - Complete Q&A Guide.md   ← source of truth
                        │
                        ▼
              docs/index.md + docs/guide.md              ← symlinks, not copies
                        │
                        ▼
                    mkdocs.yml                           ← site config
                        │
                        ▼
            GitHub Actions on push to main               ← CI/CD
                        │
                        ▼
                gh-pages branch deploy
                        │
                        ▼
              Published GitHub Pages site
```

---

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

---

## Deploy Your Own

1. Create a new GitHub repository.
2. Push this folder's contents to the `main` branch.
3. In GitHub, open **Settings → Pages**.
4. Set the source to **Deploy from a branch** and choose `gh-pages` → `/ (root)`.
5. Push to `main` — the workflow in `.github/workflows/deploy-pages.yml` will publish the site.

---

## Why This Is Different

- Focuses on **system design trade-offs**, not just definitions.
- Connects **retrieval, generation, and operations** in one continuous reference.
- Covers both **foundations** and **production realities**: evals, observability, security, and scaling.

---

## RAG Ecosystem

This repo is part of a broader RAG toolkit:

| Repo | What it covers |
| --- | --- |
| [rag-auditor](https://github.com/amitgambhir/rag-auditor) | Evaluate your RAG pipeline |
| [multi-llm-rag-agent-chat](https://github.com/amitgambhir/multi-llm-rag-agent-chat) | Build a production RAG chatbot with multi-LLM routing |
| [rag-system-design-guide](https://github.com/amitgambhir/rag-system-design-guide) ← you are here | Design reference — architecture patterns and trade-offs |

Start with the design guide, build with the chatbot, evaluate with the auditor.

---

## License

Released under the [MIT License](LICENSE).

*I wrote this as the reference I wish I had when turning RAG ideas into production architecture.*

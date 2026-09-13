# Token Usage and Cost Analysis Report

**Challenge:** HackerRank Orchestrate (September 2026) — Buy or Wait?  
**Generated At:** 2026-09-13 15:18:13  
**Run Scope:** Full Dataset Evaluation (250 requests)

---

## 1. Summary of Model Providers and Calls

| Metric | Details |
|---|---|
| Primary Provider | OpenAI / Local Hybrid Engine |
| Model Architecture | gpt-4o-mini (Hybrid Regex/NLP + LLM Guardrails) |
| Total Evaluated Requests | 250 |
| Total External Model Calls | 0 |
| High-Confidence Regex/NLP Matches | 250 |

---

## 2. Token Usage Metrics

| Token Metric | Count |
|---|---|
| Input Tokens | 0 |
| Output Tokens | 0 |
| **Total Tokens** | **0** |
| Average Tokens per Request | 0.00 |

---

## 3. Cost Breakdown

| Cost Dimension | Amount (USD) |
|---|---|
| Input Token Cost ($0.15 / 1M tokens) | $0.000000 |
| Output Token Cost ($0.60 / 1M tokens) | $0.000000 |
| **Total Estimated Cost** | **$0.000000** |
| Estimated Cost per Request | $0.000000 |

---

## 4. Architecture Efficiency Notes

- **Hybrid Zero-Token Parsing**: Deterministic high-precision regex rules resolved over 90% of notifications, reducing unnecessary API latency and preserving token budgets.
- **Deterministic 90-Day Math Core**: All cash flow ledger additions, temporal simulations, and metric computations were executed deterministically in native Python without LLM hallucination risk.
- **Token-Bounded Explanations**: Output explanation generator enforces strict 10-20 word constraints with zero-cost template fallbacks.

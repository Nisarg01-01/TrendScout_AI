# Professor's Idea (Original Notes)

## 1. Core Pipeline Concept

1. **Get data from TechCrunch.**
2. **Make a graph of the articles** based on connectivity:  
   If an article discusses startups S1, S2, S3 and another article discusses S3, S2, S8,  
   then S1 and S8 are considered somewhat connected.
3. **Cluster the article graph** using the Lumen (likely Louvain or similar) algorithm.
4. **Within each article cluster**, make a **graph of paragraphs/snippets** of the articles.  
   - These snippet graphs represent one KPI (Key Performance Indicator).  
   - Determine whether the KPI sentiment is **positive or negative** — for example,  
     “getting investment from Google” → positive; “not getting investment” → negative.
5. Use **RAG (Retrieval-Augmented Generation)** and a **Knowledge Graph** as necessary for reasoning and answering queries.
6. **Rank startups cluster-wise** and perform **KPI comparisons** between startups in the same article cluster and KPI clusters from the second graph.

---

## 2. Forecasting and Predictive Layer (Second Recommendation)

After building the main graphs and ranking system, the next recommendation is to add a **forecasting component**:  
- Predict whether a startup will **succeed or fail** (based on its trends, KPIs, and context).  
- Provide the **reason for the prediction** through an **LLM-powered chatbot** that explains using evidence from the graphs and snippets.

---

## 3. Temporal/Time Dimension (Final Recommendation)

Finally, add a **temporal aspect** to the system:  
- Record **what happened and when**.  
- Model how startup events evolve over time.  
- Use this temporal reasoning for trend detection, forecasting, and conversational explanations.

---

### Summary (Essence of the Professor’s Plan)

| Stage | Focus | Output |
|--------|--------|---------|
| 1 | Article graph of co-mentioned startups | Thematic clusters of connected startups |
| 2 | Snippet-level KPI graphs within each cluster | KPI polarity (+/-) and evidence per startup |
| 3 | Cross-graph comparison | KPI-based ranking between startups |
| 4 | Forecasting | Predict startup success/failure with explanations |
| 5 | Temporal reasoning | Add “when” dimension for events and predictions |

---

### Alignment with TrendScout Project

Your current TrendScout pipeline implements this plan with:
- Legal RSS ingestion from TechCrunch (and similar sources).  
- Article co-mention graph + Louvain clustering.  
- KPI extraction with polarity detection from snippets.  
- Neo4j-based knowledge graph for storage and retrieval.  
- Planned modules for ranking, RAG, forecasting, and temporal analytics.

This document captures the **professor’s original guidance** for long-term reference.

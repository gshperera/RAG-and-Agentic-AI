# 📚 Chunking Strategies for RAG Systems

## 📑 Table of Contents
- [What is Chunking?](#what-is-chunking)
- [Why Better Chunking Matters](#why-better-chunking-matters)
- [The 5 Chunking Strategies](#the-5-chunking-strategies)
  - [1. CharacterTextSplitter (Basic)](#1-charactertextsplitter-basic)
  - [2. RecursiveCharacterTextSplitter (Advanced)](#2-recursivecharactertextsplitter-advanced)
  - [3. Document-Specific Splting](#3-document-specific-splitting)
  - [4. Semantic Splitting (Deep Dive)](#4-semantic-splitting-deep-dive)
  - [5. Agentic Splitting (AI-Powered)](#5-agentic-splitting-ai-powered)
- [Final Recommendation](#final-recommendation)

---

## What is Chunking?

Chunking is the process of dividing a large document into smaller, manageable pieces (called *chunks*).  
In a RAG system, the retriever does **not** search entire documents – it searches chunks. The quality of those chunks directly determines:

- How accurately relevant information is retrieved
- How well the LLM can generate a coherent answer

> **Bad chunks = bad answers.**  
> Even perfect embeddings cannot fix poorly split content.

---

## Why Chunking Matters

| Problem with bad chunking | Consequence |
|---------------------------|-------------|
| **Too small** | Lacks context → retrieval misses connections |
| **Too large** | Too much noise → exceeds embedding/context window limits |
| **Poor boundaries** | Splits related information across chunks |
| **No structure awareness** | Ignores document format (headers, lists, code blocks) |

---

## 🧠 The 5 Chunking Strategies
| Strategy | Approach | Best for |
|----------|----------|----------|
| [CharacterTextSplitter](#1-charactertextsplitter) | Fixed character count, custom separators | Simple, uniform documents; when speed matters most |
| [RecursiveCharacterTextSplitter](#2-recursivecharactertextsplitter) | Recursively tries natural boundaries (paragraphs, sentences, words) | General purpose; upgrade from basic splitter |
| [Document‑Specific Splitting](#3-documentspecific-splitting) | Respects document structure (PDF pages, Markdown headers) | Structured documents (PDFs, Markdown, code) |
| [Semantic Splitting](#4-semantic-splitting) | Uses embeddings to detect topic shifts | Content where meaning changes gradually |
| [Agentic Splitting](#5-agentic-splitting) | LLM analyses and decides optimal splits | Complex documents, maximum quality (slowest) |

---

## 1. CharacterTextSplitter

### How it works
 
It follows a **split-first, merge-second** approach:
 
1. **Split** — Break the text at a defined separator (default: `\n\n`).
2. **Merge** — Combine pieces until the chunk size hits the `chunk_size` limit.
 
### Example configuration
 
```python
from langchain_text_splitters import CharacterTextSplitter
 
splitter = CharacterTextSplitter(
    chunk_size=100,      # characters
    separator="\n\n"     # default separator
)
```
 
### Example with Tesla text
 
**Input text:**
 
```
Tesla's Q3 Results
 
Tesla reported record revenue of $25.28 in Q3 2024.
 
Model Y Performance
 
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
 
Production Challenges
 
Supply chain issues caused a 12% increase in production costs.
```
 
**Step 1 — Split at `\n\n`:**
 
| Piece | Content | Length |
|-------|---------|--------|
| 1 | "Tesla's Q3 Results" | 18 chars |
| 2 | "Tesla reported record revenue of $25.28 in Q3 2024." | 51 chars |
| 3 | "Model Y Performance" | 19 chars |
| 4 | "The Model Y became the best-selling vehicle globally, with 350,000 units sold." | 78 chars |
| 5 | "Production Challenges" | 21 chars |
| 6 | "Supply chain issues caused a 12% increase in production costs." | 62 chars |
 
**Step 2 — Merge until `chunk_size=100`:**
 
- **Chunk 1** (92 chars): `"Tesla's Q3 Results\n\nTesla reported record revenue of $25.28 in Q3 2024.\n\nModel Y Performance"`
- **Chunk 2** (78 chars): `"The Model Y became the best-selling vehicle globally, with 350,000 units sold."`
- **Chunk 3** (85 chars): `"Production Challenges\n\nSupply chain issues caused a 12% increase in production costs."`
 
### ⚠️ The Problem
 
If a single piece is already larger than `chunk_size` and contains no separator, `CharacterTextSplitter` cannot split it further.
 
```python
text = "This sentence is written to be deliberately long, flowing, and continuous, \
without breaking into shorter sections. This nearly reaches two hundred characters in total length overall."
# No "\n\n" exists → the splitter returns the whole text as one chunk, exceeding the limit.
```
 
### ✅ Solution: RecursiveCharacterTextSplitter

---

## 2. RecursiveCharacterTextSplitter
 
### How it works
 
Instead of relying on a single separator, it tries a list of separators in order:
 
1. Try `"\n\n"` (paragraph breaks)
2. If still too large, try `"\n"` (line breaks)
3. Then try `" "` (word boundaries)
4. Finally fall back to character-level splitting
 
This ensures that no chunk exceeds `chunk_size` while preserving as much natural language structure as possible.
 
### Example
 
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""]
)
 
chunks = splitter.split_text(long_sentence)
# Result: The long sentence is split at word boundaries into multiple chunks of ≤100 chars.
```
 
### Why it's an upgrade
 
- Preserves more context than basic splitting
- Graceful fallback — never returns an oversized chunk
- Works well for most general-purpose documents
 
> **Limitation:** Still does not understand document structure (headers, lists, tables). For that, use document-specific splitting.

---

## 3. Document-Specific Splitting
 
### How it works
 
Different document types have built-in structural hints. A good splitter respects those:
 
| Document type | Splitting strategy |
|---------------|-------------------|
| PDF | Pages, sections, headers, footnotes |
| Markdown | Headers (`#`, `##`, `###`), code blocks (` ``` `), lists |
| HTML | `<h1>`…`<h6>`, `<div>` classes, `<section>` |
| Code | Functions, classes, import blocks |
 

 
### Benefits
 
- Keeps related information (e.g., a code block) intact
- Preserves hierarchical context
- Avoids splitting inside a table or list

---

## 4. Semantic Splitting
 
### How it works
 
Instead of using fixed character counts or separators, semantic chunking uses embeddings to detect where the topic changes naturally.
 
**3-Step process:**

1. **Encode** — Convert each sentence into a vector embedding.
2. **Compare** — Calculate similarity scores between consecutive sentences (cosine similarity).
3. **Split** — Create a boundary where the similarity drops significantly.
    The most common breakpont criteria to decide semantic splitting with SemanticChunker is with `percentile`
 
### What is a percentile?

A percentile tells you what percentage of values are below a certain number.
If you're at the 70th percentile, it means 70% of all values are below yours, and only 30% are above yours.

**Example:** 
If 200,000 students took an exam and you scored at the 95th percentile, it means:
- You scored better than 190,000 students (95% of all text takers)
- Only 10,000 students (5%) scored higher than you
- You're in the top 5% of all test takers

Percentiles are all about relative ranking, not absolute scores. The same score can be great or poor depending on how others performed

### How to we calculate the percentile?
 
1. Collect all similarity scores between consecutive sentences.
2. Sort them from lowest to highest.
3. Choose a percentile (commonly 70th).
4. Find the score at that percentile — that becomes the split threshold.
 
**Example with Tesla text:**
 
- Similarity scores: `[0.85, 0.78, 0.42, 0.71, 0.95]`
- Sorted: `[0.42, 0.71, 0.78, 0.85, 0.95]`
- Pick your percentile: `(40th percentile)`
- 40th percentile of 5 scores = position `0.40 × 5 = 2nd position`
- Find the score at 2nd position: 0.71 `(40th percentile = 0.71)`
- **Split rule:** split wherever similarity ≤ 0.71
 
| Pair | Similarity | Action |
|------|------------|--------|
| 1–2 | 0.85 | keep together |
| 2–3 | 0.78 | keep together |
| 3–4 | 0.42 | **SPLIT** (topic shift: revenue → vehicle sales) |
| 4–5 | 0.71 | keep together |

**Result:** Split after sentence 3 (the topic shift from revenue to vehicle sales)

### Why use Percentile?
 
A fixed similarity threshold (e.g., `0.70`) fails when documents have different similarity patterns:
 
| Document type | Typical similarities | Fixed 0.70 threshold |
|---------------|---------------------|----------------------|
| Academic paper | High `[0.85, 0.88, 0.91, 0.87, 0.89]` | Never splits (all >0.70) ❌ |
| News article | Lower `[0.45, 0.52, 0.38, 0.61, 0.43]` | Splits everywhere (all <0.70) ❌ |
 
**Percentile** solves this by adapting to each document's own distribution.
- Academic paper: 70th percentile = 0.88
- News articles: 70th percentile = 0.52
 
### Why 70th percentile?
 
- Not too aggressive — won't split at every small dip
- Not too conservative — won't miss obvious topic changes
- Works well for most documents (range: 60th–90th percentile)
 

 
### Trade-offs
 
| | |
|--|--|
| ✅ **Pro** | Content-aware, adapts to each document, preserves meaning |
| ❌ **Con** | Computationally expensive (embeddings for every sentence), slower than rule-based methods |

---

## 5. Agentic Splitting
 
### How it works
 
An LLM (e.g., GPT-3.5/4) acts as a "chunking agent" that reads the document and decides where the optimal splits are.
 
The agent can:
- Understand complex relationships across paragraphs
- Respect implicit topic boundaries (e.g., a story's scene change)
- Adapt to any content type without pre-configured rules
 
### Implementation example
 
```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
 
load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
 
tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.
 
Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.
 
Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""
 
prompt = f"""
You are a text chunking expert. Split this text into logical chunks.
 
Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks
 
Text:
{tesla_text}
 
Return the text with <<<SPLIT>>> markers where you want to split:
"""
 
response = llm.invoke(prompt)
marked_text = response.content

```

### Example output
 
```
Tesla's Q3 Results
<<<SPLIT>>>
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.

<<<SPLIT>>>
Revenue growth was driven by strong vehicle deliveries.
Model Y Performance
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

<<<SPLIT>>>
Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs.

```
 
### Advantages
 
- **Maximum quality** — splits are semantically perfect for the specific document
- **Zero configuration** — no separators, no chunk size, no percentiles to tune
- **Can handle mixed content** (narrative + data + code)
 
### Disadvantages
 
- **Slow** — each document requires one or more LLM calls
- **Expensive** — cost scales with document length and LLM size
- **Non-deterministic** — same document may produce slightly different splits each time (`temperature=0` reduces this)
- **Not real-time friendly** — unsuitable for low-latency pipelines
 
### When to use agentic splitting
 
- Offline indexing of high-value documents (legal, medical, technical)
- When retrieval quality is the top priority and budget allows
- For documents where rule-based splitters consistently fail
 
---
 
## Choosing the Right Strategy
 
| If you need… | Use… |
|--------------|-------|
| Speed & simplicity | CharacterTextSplitter |
| General purpose, no oversize chunks | RecursiveCharacterTextSplitter |
| Respect for document format (PDF, Markdown) | Document-Specific Splitting |
| Content-aware boundaries, adaptive | Semantic Splitting (percentile) |
| Maximum accuracy, offline processing | Agentic Splitting |
 
### Typical pipeline recommendation
 
1. **Start with `RecursiveCharacterTextSplitter`** — it works well for most use cases.
2. If your documents have clear structure (headers, tables), **add document-specific splitting**.
3. If retrieval still misses obvious topic shifts, **try Semantic Splitting**.
4. Only for critical, high-value documents, **invest in Agentic Splitting**.
 
---
 
## Summary
 
- Chunking is not an afterthought — it is an **important step in RAG**.
- **Basic splitters** (`CharacterTextSplitter`) are simple but fragile.
- **Recursive splitters** fix the "oversized piece" problem.
- **Document-aware splitters** preserve format.
- **Semantic splitters** use embeddings and adaptive thresholds (percentile) to follow meaning.
- **Agentic splitters** leverage LLMs for near-perfect chunking at higher cost.
 
---
 
> 🎯 **Remember:** Your retriever searches chunks, not documents. Make every chunk count.
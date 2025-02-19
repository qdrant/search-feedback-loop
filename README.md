# Search Feedback Loop
## Idea:
Teach agents to use Discovery API instead of a bad query reformulation.
## Goal:
Better Agentic RAG results.

## Basic Experiment:
- `all-MiniLM-L6-v2` as main model;
- `mxbai-embed-large-v1` as agent;
- BEIR datasets for eval;
- comparing top-1 hits (only top-1 matters for RAG) for each query: `hits/queries_amount`

### Expensive agent scenario
top-10 results of `all-MiniLM-L6-v2` reranked with `mxbai-embed-large-v1`

### Discovery-aware agent scenario
1. top-3 results of `all-MiniLM-L6-v2` reranked with `mxbai-embed-large-v1`
If any results in top-3 changed their order, we have feedback from the agent -- context for discovery
2. Discovery with `positive context` (top-1 reranked) and `negative context` (top-3 reranked) using `all-MiniLM-L6-v2`, results from 1. excluded
3. Reranking discovered top-3 with `mxbai-embed-large-v1`
Selecting the best top-1 result from 1 and 3 based on the `mxbai-embed-large-v1` score.

## How to run
BEIR datasets folders should be downloaded and put on the same level as scripts;
In the current set-up, `Qdrant Cloud` is used, and credentials are taken from `config.ini`.
So, to use it also with Cloud, `config.ini` should be changed with your credentials.

This is the example for running scripts on `FiQa-2018.`
1. `indexing.py`

```bash
python indexing.py --dataset_path nfcorpus/corpus.jsonl --total-points-in-dataset 3600 --collection-name "discovery_agents"
```
2. `evaluating.py`

```bash
python evaluating.py --input-path-queries nfcorpus/queries.jsonl --input-path-qrels nfcorpus/qrels/test.tsv --collection-name "discovery_agents" --total-queries-in-dataset 323 
```

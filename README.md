# Intent Qualification System

A local AI system that turns natural-language company-search queries into structured requirements and returns a ranked list of candidate companies.

The pipeline combines structured query extraction, feature-specific embedding comparisons, weighted softmin aggregation, numerical filtering, and an IVF index built with K-Means.

## Objective

Retrieve companies that satisfy the user's intent, including:
- Business activity and offerings
- Customer segments and business models
- Geographic requirements
- Founding year, revenue, and employee-count constraints

## Architecture

### 1. Structured Query Extraction
- Local LFM model served through llama.cpp
- Pydantic schema for structured output
- Separation of company requirements from customer and supplier attributes

### 2. Company Representation
- Nemotron embedding model
- Eight feature groups
- Normalized, 512-dimensional embeddings
- Cached embeddings and index data

### 3. Candidate Retrieval
- K-Means clustering for each feature group
- Inverted lists containing company indices
- Selection of nearby clusters through `nprobe`
- Union of candidates across requested groups

### 4. Ranking and Filtering
- Cosine similarity between corresponding query and company feature groups
- Weighted softmin aggregation
- Explicit numerical filtering
- Top-K ranked output

## Implementation Decisions
- Why compare corresponding feature groups?
- Why use weighted softmin?
- Why keep numerical comparisons outside semantic scoring?
- Why use IVF, and what does approximation change?
- How are missing fields handled?

## Installation
[Environment, dependencies, model downloads, and hardware requirements]

## Usage
[Configuration, llama-server startup, index construction, and query execution]

## Evaluation
- Twelve evaluation queries
- Repeated runs
- Separate extraction and search latency measurements
- Saved structured outputs and ranked results
- IVF comparison against exhaustive search, when available

## Results
[Measured latency, retrieval fidelity, and representative examples]

## Limitations
- Extraction variability
- Missing or inconsistent company data
- Approximate retrieval may omit candidates
- Semantic similarity does not establish that every requirement is satisfied
- Performance at larger scales requires separate benchmarking

## Reproducibility
[Seeds, model versions, hyperparameters, index configuration, and evaluation commands]

## Repository Structure
[Main pipeline, evaluation scripts, configuration, and output locations]

# Intent Qualification System

A local AI system that turns natural-language company-search queries into structured requirements and returns a ranked list of candidate companies.

The pipeline combines structured query extraction, feature-specific embedding comparisons, weighted softmin aggregation, numerical filtering, and an IVF index built with K-Means.

## Objective

The scope of this project is to retrieve companies that satisfy the user's intent, including:
- Business activity and offerings
- Customer segments and business models
- Geographic requirements
- Founding year, revenue, and employee-count constraints
- Additional information regarding website url, country code, etc.

## Architecture

### Structured Query Extraction
Local AI model (LFM 3 1.7B in Q4) served through llama.cpp processes the input query and provides structured output via Pydantic with separation of company requirements via its feature classes: 
  - Identity — listing status, website URL
  - Industry — industry type
  - Description — company description
  - Core offerings — products and services
  - Business model — business models
  - Target markets — customer segments and industries served
  - Geography — country or region, town or city
  - Structured — founding year, employee count, annual revenue
  

### Company Database Representation
The JSONL database is embedded using the Nemotron 3.5  1B ( in bf16) embedding model for each company class, resulting in a normalized embedding vector of size 8 x 512. 
For scalability purposes and reduced semantic search latency, the vector embeddings were grouped with K means algorithm, for each feature class, with automatic selection of the adequate number of centroids through the maximum distance from the line given by the maximum and minimum values ( Elbow Method). After this each cluster is stored as an inverted list containing the indices of its assigned embedding vectors.

### Candidate Retrieval, Ranking and Filtering
The candidates were retrived by first checking the closest n clusters ( minimum L2 distance) and extracting the representations of each selected cluster, then the dot product between the query embedding and each company embbeding was done for each feature class, dropping missing data from the search ( set with 0 similarity). 
This was followed by weighted softmin aggregation with predefined weights to reduce the risk of strong matches on some features masking weak matches on others. I selected the weigths as "identity" = 5%, "industry" = 15%, "description" = 10%, "core_offerings" = 25%, "business_model" = 15%, "target_markets" = 20%, "geography" = 5%, "structured" = 5%. 
At the end, numerical filtering was applied for the founding year, revenue  and employee count  ( greater, under, equal), followed by the selection of the top 10 candidates.

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

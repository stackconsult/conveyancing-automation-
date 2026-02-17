# Stage 1 Retrieval System Specification

This directory contains the complete Stage 1 Intelligent Retrieval system specification for the conveyancing automation platform.

## Overview

Stage 1 provides the critical bridge between the beautifully chunked documents from Stage 0 and the DeepSeek-R1 reasoning engine. It ensures that downstream agents (Investigator, Tax, Scribe) always receive the right slices of 100-page+ Alberta conveyancing files without facing the risk vs coverage tradeoff.

## Architecture Components

### Core Components
- **PreProcessingAgent** - OCR, chunking, Mem0 registry (Stage 0)
- **RetrievalIntent Model** - Standardized intent schema
- **Chunk Registry & Index** - Mem0 + vector database
- **SegmentAwareRetriever** - Hybrid retrieval logic
- **Risk-Aware Ranker** - Risk-weighted scoring engine
- **ContextPackager** - Bounded context assembly
- **CoverageSelfCheck** - Structural coverage validation
- **Error & Fallback Layer** - Deterministic failure handling
- **Telemetry & Audit Logging** - Comprehensive tracking

### Document Types Supported
- Long purchase agreements / builder contracts
- Multi-title searches and certificate of title packages
- Condo document bundles (bylaws, minutes, budgets, reserve fund reports)
- Additional Alberta conveyancing documents (future phases)

## Key Features

- **No risk vs coverage tradeoff** - Multi-stage retrieval that maximizes expected risk coverage per token
- **Mem0 integration** - 4-layer memory architecture (deal, agent, global, meta)
- **Alberta-specific** - Tailored to Alberta conveyancing workflows and document types
- **Production-ready** - Minimal components, explicit contracts, full auditability
- **DeepSeek-R1 optimized** - Segment-aware context packaging

## Implementation Status

- ✅ Stage 0: Pre-processing & Preparation (Complete)
- 🔄 Stage 1: Intelligent Retrieval (In Progress)
- ⏳ Stage 2: R1 Reasoning Passes (Pending)
- ⏳ Stage 3: Synthesizer Agent (Pending)
- ⏳ Stage 4: Human Gate UX (Pending)

## Files Structure

```
stage1_retrieval/
├── README.md                    # This overview
├── specification.md             # Complete technical specification
├── schemas/                     # Pydantic models and contracts
│   ├── primitives.py           # Base types and enums
│   ├── document_models.py      # DocumentChunk and related models
│   ├── retrieval_models.py     # RetrievalIntent and result models
│   └── context_models.py       # ContextPackage and summary models
├── algorithms/                  # Core retrieval algorithms
│   ├── segment_aware_retriever.py
│   ├── risk_aware_ranker.py
│   ├── context_packager.py
│   └── coverage_self_check.py
├── integration/                 # LangGraph and agent integration
│   ├── retrieval_agent.py       # Main RetrievalAgent class
│   ├── langgraph_node.py        # LangGraph node interface
│   └── agent_adapters.py        # Investigator/Tax/Scribe adapters
├── tests/                       # Comprehensive test suite
│   ├── test_schemas.py
│   ├── test_algorithms.py
│   └── test_integration.py
└── examples/                    # Usage examples and demos
    ├── alberta_deal_example.py
    └── retrieval_demo.py
```

## Next Steps

1. Review the complete specification in `specification.md`
2. Implement schemas and contracts in `schemas/`
3. Build core algorithms in `algorithms/`
4. Create integration hooks in `integration/`
5. Add comprehensive tests in `tests/`
6. Deploy and validate with Alberta conveyancing documents

## Technical Requirements

- Python 3.11+
- Mem0 Platform API
- Vector database (Qdrant/Weaviate)
- Azure Document Intelligence
- DeepSeek-R1 endpoint
- LangGraph orchestration

## Documentation

See the complete technical specification for detailed implementation guidance, API contracts, and integration patterns.

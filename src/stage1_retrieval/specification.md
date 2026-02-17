# Stage 1 Retrieval System - Complete Technical Specification

## Executive Summary

Stage 1 Retrieval provides the intelligent document slicing layer that bridges Stage 0 preprocessing with DeepSeek-R1 reasoning. It eliminates the risk vs coverage tradeoff through multi-stage retrieval that maximizes expected risk coverage per token while maintaining strict auditability.

## 1. System Architecture

### 1.1 Macro Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  PreProcessing  │───▶│  Chunk Registry  │───▶│  RetrievalAgent │
│     Agent       │    │   & Index        │    │   (Orchestrator)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                       ┌──────────────────────────────┼──────────────────────────────┐
                       │                              │                              │
            ┌──────────▼──────────┐        ┌─────────▼─────────┐        ┌────────▼────────┐
            │ SegmentAwareRetriever│        │ Risk-Aware Ranker │        │ContextPackager │
            └─────────────────────┘        └───────────────────┘        └─────────────────┘
                       │                              │                              │
                       └──────────────────────────────┼──────────────────────────────┘
                                                      │
                                      ┌─────────────────▼─────────────────┐
                                      │     CoverageSelfCheck & Fallback   │
                                      └───────────────────────────────────┘
```

### 1.2 Data Flow

1. **Input**: RetrievalIntent from downstream agents (Investigator, Tax, Scribe)
2. **Filtering**: Structural filtering via Mem0 metadata
3. **Retrieval**: Semantic search over filtered chunks
4. **Ranking**: Risk-weighted re-ranking
5. **Packaging**: Bounded context assembly
6. **Validation**: Coverage self-check
7. **Output**: ContextPackage + RetrievalSummary

## 2. Core Data Models

### 2.1 Primitives

```python
# Scalar primitives
DealId = str                    # UUID or {year}-{firm}-{sequential}
DocumentId = str                 # Per-file identifier
ChunkId = str                    # sha256("{DocumentId}:{page_start}-{page_end}:{content_hash}")
UserId = str                     # Mem0 scope (conv-{DealId})
AgentId = str                    # "preprocessing_v1", "retrieval_v1", "investigator_r1"

# Enums
class Stage(str, Enum):
    INTAKE = "INTAKE"
    DILIGENCE = "DILIGENCE"
    DRAFTING = "DRAFTING"
    CLOSING = "CLOSING"

class DocumentRole(str, Enum):
    PURCHASE_AGREEMENT = "PURCHASE_AGREEMENT"
    TITLE_SEARCH = "TITLE_SEARCH"
    CONDO_DOCS = "CONDO_DOCS"
    TAX_CERTIFICATE = "TAX_CERTIFICATE"
    LENDER_INSTRUCTIONS = "LENDER_INSTRUCTIONS"
    TRANSFER_OF_LAND = "TRANSFER_OF_LAND"
    STATEMENT_OF_ADJUSTMENTS = "STATEMENT_OF_ADJUSTMENTS"
    OTHER = "OTHER"

class SectionType(str, Enum):
    TITLE_SUMMARY = "TITLE_SUMMARY"
    INSTRUMENTS_REGISTER = "INSTRUMENTS_REGISTER"
    CAVEATS_SECTION = "CAVEATS_SECTION"
    LEGAL_DESCRIPTION = "LEGAL_DESCRIPTION"
    CONDO_BYLAWS = "CONDO_BYLAWS"
    CONDO_MINUTES = "CONDO_MINUTES"
    RESERVE_FUND_REPORT = "RESERVE_FUND_REPORT"
    FINANCIAL_STATEMENTS = "FINANCIAL_STATEMENTS"
    SPECIAL_RESOLUTIONS = "SPECIAL_RESOLUTIONS"
    TAX_ARREARS = "TAX_ARREARS"
    GENERAL = "GENERAL"

class RiskProfile(str, Enum):
    HIGH_RISK = "HIGH_RISK"
    BALANCED = "BALANCED"
    LOW_RISK = "LOW_RISK"
```

### 2.2 DocumentChunk Model

```python
@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    deal_id: str
    document_id: str
    document_role: DocumentRole
    page_start: int
    page_end: int
    section_type: SectionType
    content: str
    token_count: int
    ocr_confidence: float
    content_hash: str
    metadata: Dict[str, Any]
    
    # Computed properties
    @property
    def page_range(self) -> str:
        return f"{self.page_start}-{self.page_end}"
    
    @property
    def is_high_confidence(self) -> bool:
        return self.ocr_confidence >= 0.85
```

### 2.3 RetrievalIntent Model

```python
@dataclass(frozen=True)
class RetrievalIntent:
    intent_id: str                    # e.g., "title_risk_scan_v1"
    deal_id: str
    agent_id: str
    query_text: str
    target_section_types: List[SectionType]
    risk_profile: RiskProfile
    max_tokens_budget: int
    required_structural_zones: Optional[List[str]] = None
    
    # Validation
    def __post_init__(self):
        assert self.max_tokens_budget > 0
        assert len(self.target_section_types) > 0
        assert self.deal_id and self.agent_id
```

### 2.4 ContextPackage Model

```python
@dataclass(frozen=True)
class ContextPackage:
    deal_id: str
    intent_id: str
    agent_id: str
    ordered_chunks: List[ChunkReference]
    structural_toc: str
    exclusions_note: str
    total_tokens: int
    risk_summary: str
    
    @dataclass(frozen=True)
    class ChunkReference:
        chunk_id: str
        page_range: str
        section_type: SectionType
        content: str
        content_hash: str
        document_role: DocumentRole
```

## 3. Algorithm Specifications

### 3.1 SegmentAwareRetriever

**Objective**: Retrieve structurally coherent candidates using hybrid search.

**Algorithm**:
1. **Structural Filtering**: Query Mem0 with deal_id + section_type filters
2. **Semantic Search**: Vector search over filtered candidates
3. **Coherence Grouping**: Group adjacent page ranges
4. **Candidate Selection**: Return top-K candidates with structural metadata

**Complexity**: O(N) filtering + O(log M) vector search where N = chunks in deal, M = total indexed chunks

### 3.2 Risk-Aware Ranker

**Objective**: Re-rank candidates using risk heuristics and historical patterns.

**Risk Signals**:
- **Title**: "caveat", "builder's lien", "homeowners association", "special assessment"
- **Condo**: "special levy", "structural deficiency", "reserve fund shortfall"
- **Tax**: "arrears", "penalty", "lien", "installment plan"

**Scoring Formula**:
```
combined_score = α * semantic_score + β * risk_signal_score + γ * mem0_pattern_score
```

Where α, β, γ weights depend on RiskProfile:
- HIGH_RISK: α=0.3, β=0.5, γ=0.2
- BALANCED: α=0.5, β=0.3, γ=0.2  
- LOW_RISK: α=0.7, β=0.2, γ=0.1

### 3.3 CoverageSelfCheck

**Objective**: Ensure structurally important zones are not missed.

**Required Zones per Intent**:
```python
REQUIRED_ZONES = {
    "title_risk_scan_v1": ["front_summary", "instruments_register", "caveats_section"],
    "condo_reserve_fund_health_v1": ["reserve_fund_report", "financial_statements", "special_resolutions"],
    "tax_arrears_check_v1": ["tax_certificate", "municipal_arrears", "tax_liens"]
}
```

**Algorithm**:
1. Compare ContextPackage.structural_toc against required zones
2. Identify missing/partial zones
3. Trigger targeted retrieval for gaps
4. Merge supplementary chunks
5. Update exclusions note

## 4. Mem0 Integration

### 4.1 Memory Architecture

```
Layer 1: Per-deal memory
  Pattern: deal/{deal_id}/{topic}
  Scope: Deal-specific state, documents, risks, waivers

Layer 2: Per-agent operational memory  
  Pattern: agent/{agent_name}/{capability}/{version}
  Scope: Agent tactics, prompt improvements

Layer 3: Global institutional memory
  Pattern: global/{entity_type}/{entity_id}/{topic}
  Scope: Cross-file patterns, repeat offenders

Layer 4: Meta-methods memory
  Pattern: meta/{component}/{method}
  Scope: System learning, pattern updates
```

### 4.2 Memory Operations

**Write Operations**:
```python
# Chunk registry
await mem0.add(
    messages=[{"role": "system", "content": chunk_metadata}],
    user_id=f"conv-{deal_id}",
    metadata={
        "chunk_id": chunk_id,
        "section_type": section_type,
        "page_range": page_range,
        "ocr_confidence": ocr_confidence,
        "content_hash": content_hash,
        "document_role": document_role,
        "agent_id": "preprocessing_v1"
    }
)

# Retrieval audit
await mem0.add(
    messages=[{"role": "system", "content": retrieval_summary}],
    user_id=f"conv-{deal_id}",
    metadata={
        "intent_id": intent_id,
        "agent_id": "retrieval_v1",
        "chunks_inspected": len(candidates),
        "chunks_selected": len(selected),
        "risk_distribution": risk_summary
    }
)
```

**Read Operations**:
```python
# Structural filtering
chunks = await mem0.search(
    query="",
    version="v2",
    filters={
        "AND": [
            {"user_id": f"conv-{deal_id}"},
            {"section_type": {"in": target_section_types}}
        ]
    }
)

# Pattern retrieval
patterns = await mem0.search(
    query="builder lien issues",
    version="v2", 
    filters={"OR": [{"entity_id": builder_id}, {"entity_type": "builder"}]}
)
```

## 5. Error Handling & Fallbacks

### 5.1 Failure Modes

| Failure Mode | Detection | Response |
|-------------|-----------|----------|
| No chunks available | Empty Mem0 results | needs_human_input flag |
| Token budget insufficient | Budget exceeded during packaging | Reduce scope or escalate |
| Mem0 unavailable | Connection/timeout errors | Fallback to vector DB only |
| Low confidence results | All scores < threshold | Broaden search or human review |

### 5.2 Fallback Strategy

```python
class RetrievalFallback:
    @staticmethod
    async def handle_no_chunks(intent: RetrievalIntent) -> ContextPackage:
        # Return minimal package with explanation
        return ContextPackage(
            deal_id=intent.deal_id,
            intent_id=intent.intent_id,
            agent_id=intent.agent_id,
            ordered_chunks=[],
            structural_toc="No documents available",
            exclusions_note=f"No {intent.target_section_types} documents found",
            total_tokens=0,
            risk_summary="UNABLE_TO_ASSESS"
        )
    
    @staticmethod
    async def handle_budget_exhaustion(
        candidates: List[DocumentChunk], 
        budget: int
    ) -> List[DocumentChunk]:
        # Greedy selection preserving structural coverage
        selected = []
        remaining_budget = budget
        
        # Prioritize high-risk sections
        for chunk in sorted(candidates, key=lambda c: c.risk_score, reverse=True):
            if chunk.token_count <= remaining_budget:
                selected.append(chunk)
                remaining_budget -= chunk.token_count
        
        return selected
```

## 6. Performance & Telemetry

### 6.1 Metrics Collection

```python
@dataclass
class RetrievalMetrics:
    deal_id: str
    intent_id: str
    agent_id: str
    
    # Performance metrics
    total_latency_ms: int
    mem0_query_time_ms: int
    vector_search_time_ms: int
    ranking_time_ms: int
    packaging_time_ms: int
    
    # Quality metrics
    chunks_inspected: int
    chunks_selected: int
    tokens_selected: int
    risk_distribution: Dict[str, int]
    structural_coverage_score: float
    
    # Cost metrics
    embedding_api_calls: int
    mem0_api_calls: int
    estimated_cost_usd: float
```

### 6.2 Audit Trail

```python
@dataclass
class RetrievalAuditEntry:
    timestamp: datetime
    deal_id: str
    intent_id: str
    agent_id: str
    
    # Input state
    query_text: str
    target_sections: List[str]
    risk_profile: str
    max_tokens: int
    
    # Processing state
    candidates_found: int
    candidates_filtered: int
    final_chunks: int
    
    # Output state
    context_package_hash: str
    exclusions: List[str]
    risk_flags: List[str]
    
    # Quality indicators
    confidence_score: float
    coverage_completeness: float
    human_review_required: bool
```

## 7. Integration Points

### 7.1 LangGraph Node Interface

```python
class RetrievalNode:
    def __init__(self, retrieval_agent: RetrievalAgent):
        self.retrieval_agent = retrieval_agent
    
    async def __call__(
        self, 
        state: DealState
    ) -> DealState:
        # Extract intent from current state
        intent = self._build_intent_from_state(state)
        
        # Execute retrieval
        context_package, summary = await self.retrieval_agent.retrieve(intent)
        
        # Update state
        state.context_packages[state.current_step] = context_package
        state.retrieval_summaries[state.current_step] = summary
        
        return state
    
    def _build_intent_from_state(self, state: DealState) -> RetrievalIntent:
        # Agent-specific intent building logic
        if state.current_agent == "investigator":
            return RetrievalIntent(
                intent_id="title_risk_scan_v1",
                deal_id=state.deal_id,
                agent_id="investigator_r1",
                query_text="Identify title risks and encumbrances",
                target_section_types=[SectionType.TITLE_SUMMARY, SectionType.INSTRUMENTS_REGISTER, SectionType.CAVEATS_SECTION],
                risk_profile=RiskProfile.HIGH_RISK,
                max_tokens_budget=4000
            )
        # ... other agents
```

### 7.2 Agent Adapters

```python
class InvestigatorAdapter:
    @staticmethod
    def build_intent(state: DealState) -> RetrievalIntent:
        return RetrievalIntent(
            intent_id="title_risk_scan_v1",
            deal_id=state.deal_id,
            agent_id="investigator_r1", 
            query_text="Scan for title risks, encumbrances, and ownership issues",
            target_section_types=[
                SectionType.TITLE_SUMMARY,
                SectionType.INSTRUMENTS_REGISTER, 
                SectionType.CAVEATS_SECTION,
                SectionType.LEGAL_DESCRIPTION
            ],
            risk_profile=RiskProfile.HIGH_RISK,
            max_tokens_budget=4000,
            required_structural_zones=["front_summary", "instruments_register", "caveats_section"]
        )

class TaxAdapter:
    @staticmethod
    def build_intent(state: DealState) -> RetrievalIntent:
        return RetrievalIntent(
            intent_id="tax_arrears_check_v1",
            deal_id=state.deal_id,
            agent_id="tax_r1",
            query_text="Check for tax arrears, penalties, and municipal charges",
            target_section_types=[SectionType.TAX_CERTIFICATE, SectionType.TAX_ARREARS],
            risk_profile=RiskProfile.BALANCED,
            max_tokens_budget=2000
        )
```

## 8. Testing Strategy

### 8.1 Unit Tests

- **Model Validation**: Schema validation, edge cases
- **Algorithm Logic**: Ranking formulas, coverage checks
- **Error Handling**: All failure modes and fallbacks

### 8.2 Integration Tests

- **End-to-End Retrieval**: Full pipeline with test documents
- **Mem0 Integration**: Memory scoping, audit logging
- **LangGraph Integration**: Node execution, state management

### 8.3 Performance Tests

- **Large Document Handling**: 100+ page documents
- **Concurrent Retrieval**: Multiple simultaneous requests
- **Memory Efficiency**: Token budget enforcement

## 9. Deployment Considerations

### 9.1 Dependencies

- **Core**: Python 3.11+, asyncio, pydantic
- **Memory**: Mem0 Python client
- **Vector**: Qdrant/Weaviate client
- **OCR**: Azure Document Intelligence SDK
- **ML**: Sentence transformers for embeddings

### 9.2 Configuration

```python
@dataclass
class RetrievalConfig:
    # Vector database
    vector_db_url: str
    vector_db_api_key: str
    
    # Mem0
    mem0_api_key: str
    mem0_project_id: str
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Retrieval parameters
    default_max_tokens: int = 4000
    semantic_top_k: int = 50
    risk_threshold: float = 0.3
    
    # Performance
    cache_ttl_seconds: int = 3600
    max_concurrent_retrievals: int = 10
```

### 9.3 Monitoring

- **Health Checks**: Component connectivity, performance thresholds
- **Metrics**: Retrieval latency, accuracy, cost tracking
- **Alerting**: High error rates, budget overruns

## 10. Next Steps

1. **Implement Schemas**: Create all Pydantic models
2. **Build Algorithms**: Implement core retrieval logic
3. **Create Integration**: LangGraph nodes and agent adapters
4. **Add Tests**: Comprehensive test suite
5. **Deploy & Validate**: Production deployment with Alberta documents

This specification provides the complete technical foundation for implementing Stage 1 Retrieval that bridges document preprocessing with DeepSeek-R1 reasoning while maintaining auditability and performance.

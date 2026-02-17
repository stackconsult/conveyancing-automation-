"""
Stage 1 Retrieval System - Core Schemas and Contracts

This module contains all the foundational data models and contracts
for the Stage 1 Intelligent Retrieval system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
import hashlib

# ============================================================================
# PRIMITIVES AND ENUMS
# ============================================================================

# Scalar primitives (type aliases for clarity)
DealId = str                    # UUID or {year}-{firm}-{sequential}
DocumentId = str                 # Per-file identifier  
ChunkId = str                    # sha256("{DocumentId}:{page_start}-{page_end}:{content_hash}")
UserId = str                     # Mem0 scope (conv-{DealId})
AgentId = str                    # "preprocessing_v1", "retrieval_v1", "investigator_r1"

class Stage(str, Enum):
    """Conveyancing workflow stages"""
    INTAKE = "INTAKE"
    DILIGENCE = "DILIGENCE"
    DRAFTING = "DRAFTING"
    CLOSING = "CLOSING"

class DocumentRole(str, Enum):
    """Document types in Alberta conveyancing"""
    PURCHASE_AGREEMENT = "PURCHASE_AGREEMENT"
    TITLE_SEARCH = "TITLE_SEARCH"
    CONDO_DOCS = "CONDO_DOCS"
    TAX_CERTIFICATE = "TAX_CERTIFICATE"
    LENDER_INSTRUCTIONS = "LENDER_INSTRUCTIONS"
    TRANSFER_OF_LAND = "TRANSFER_OF_LAND"
    STATEMENT_OF_ADJUSTMENTS = "STATEMENT_OF_ADJUSTMENTS"
    OTHER = "OTHER"

class SectionType(str, Enum):
    """Section types within documents"""
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
    """Risk tolerance profiles for retrieval"""
    HIGH_RISK = "HIGH_RISK"
    BALANCED = "BALANCED"
    LOW_RISK = "LOW_RISK"

class RetrievalStatus(str, Enum):
    """Retrieval operation status"""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    NEEDS_HUMAN_INPUT = "NEEDS_HUMAN_INPUT"

# ============================================================================
# DOCUMENT MODELS
# ============================================================================

@dataclass(frozen=True)
class DocumentChunk:
    """Immutable chunk of document content with metadata"""
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate chunk data"""
        assert self.page_start >= 0
        assert self.page_end >= self.page_start
        assert 0.0 <= self.ocr_confidence <= 1.0
        assert self.token_count > 0
        assert len(self.content.strip()) > 0
    
    @property
    def page_range(self) -> str:
        """Get page range as string"""
        return f"{self.page_start}-{self.page_end}"
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if OCR confidence is high"""
        return self.ocr_confidence >= 0.85
    
    @property
    def is_multi_page(self) -> bool:
        """Check if chunk spans multiple pages"""
        return self.page_end > self.page_start
    
    @classmethod
    def create_chunk(
        cls,
        deal_id: str,
        document_id: str,
        document_role: DocumentRole,
        page_start: int,
        page_end: int,
        section_type: SectionType,
        content: str,
        ocr_confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "DocumentChunk":
        """Create a new chunk with generated ID and hash"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        chunk_id = hashlib.sha256(
            f"{document_id}:{page_start}-{page_end}:{content_hash}".encode()
        ).hexdigest()
        
        # Simple token estimation (roughly 4 chars per token)
        token_count = len(content) // 4
        
        return cls(
            chunk_id=chunk_id,
            deal_id=deal_id,
            document_id=document_id,
            document_role=document_role,
            page_start=page_start,
            page_end=page_end,
            section_type=section_type,
            content=content,
            token_count=token_count,
            ocr_confidence=ocr_confidence,
            content_hash=content_hash,
            metadata=metadata or {}
        )

@dataclass(frozen=True)
class ChunkReference:
    """Lightweight reference to a chunk for context packages"""
    chunk_id: str
    page_range: str
    section_type: SectionType
    content: str
    content_hash: str
    document_role: DocumentRole
    ocr_confidence: float
    
    @classmethod
    def from_chunk(cls, chunk: DocumentChunk) -> "ChunkReference":
        """Create reference from full chunk"""
        return cls(
            chunk_id=chunk.chunk_id,
            page_range=chunk.page_range,
            section_type=chunk.section_type,
            content=chunk.content,
            content_hash=chunk.content_hash,
            document_role=chunk.document_role,
            ocr_confidence=chunk.ocr_confidence
        )

# ============================================================================
# RETRIEVAL MODELS
# ============================================================================

@dataclass(frozen=True)
class RetrievalIntent:
    """Intent specification for retrieval operations"""
    intent_id: str                    # e.g., "title_risk_scan_v1"
    deal_id: str
    agent_id: str
    query_text: str
    target_section_types: List[SectionType]
    risk_profile: RiskProfile
    max_tokens_budget: int
    required_structural_zones: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate intent data"""
        assert self.max_tokens_budget > 0
        assert len(self.target_section_types) > 0
        assert self.deal_id and self.agent_id
        assert len(self.query_text.strip()) > 0
    
    @property
    def is_high_risk(self) -> bool:
        """Check if this is a high-risk retrieval"""
        return self.risk_profile == RiskProfile.HIGH_RISK
    
    @property
    def target_sections_str(self) -> str:
        """Get target sections as comma-separated string"""
        return ", ".join([st.value for st in self.target_section_types])

@dataclass
class CandidateChunk:
    """Chunk candidate with scoring information"""
    chunk: DocumentChunk
    semantic_score: float
    risk_signal_score: float
    mem0_pattern_score: float
    combined_score: float
    
    def __post_init__(self):
        """Validate scoring"""
        assert 0.0 <= self.semantic_score <= 1.0
        assert 0.0 <= self.risk_signal_score <= 1.0
        assert 0.0 <= self.mem0_pattern_score <= 1.0
        assert 0.0 <= self.combined_score <= 1.0

@dataclass
class CandidateSet:
    """Set of candidate chunks with metadata"""
    chunks: List[CandidateChunk]
    ranking_features: Dict[str, Any]
    total_tokens: int
    structural_coverage_report: Dict[str, bool]
    
    @property
    def chunk_count(self) -> int:
        """Get number of candidate chunks"""
        return len(self.chunks)
    
    @property
    def high_risk_chunks(self) -> List[CandidateChunk]:
        """Get chunks with high risk signals"""
        return [c for c in self.chunks if c.risk_signal_score > 0.5]
    
    @property
    def high_confidence_chunks(self) -> List[CandidateChunk]:
        """Get chunks with high OCR confidence"""
        return [c for c in self.chunks if c.chunk.is_high_confidence]

@dataclass
class RankedContextSet:
    """Ranked set of chunks ready for packaging"""
    chunks: List[DocumentChunk]
    risk_metadata: Dict[str, Any]
    risk_summary: str
    
    @property
    def total_tokens(self) -> int:
        """Get total token count"""
        return sum(chunk.token_count for chunk in self.chunks)
    
    @property
    def risk_distribution(self) -> Dict[str, int]:
        """Get distribution of risk levels"""
        distribution = {"high": 0, "medium": 0, "low": 0}
        for chunk in self.chunks:
            # Simple risk classification based on content
            if any(keyword in chunk.content.lower() for keyword in ["caveat", "lien", "deficiency"]):
                distribution["high"] += 1
            elif any(keyword in chunk.content.lower() for keyword in ["agreement", "terms", "conditions"]):
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        return distribution

# ============================================================================
# CONTEXT MODELS
# ============================================================================

@dataclass(frozen=True)
class ContextPackage:
    """Bounded context package for downstream agents"""
    deal_id: str
    intent_id: str
    agent_id: str
    ordered_chunks: List[ChunkReference]
    structural_toc: str
    exclusions_note: str
    total_tokens: int
    risk_summary: str
    
    def __post_init__(self):
        """Validate context package"""
        assert self.total_tokens > 0
        assert len(self.ordered_chunks) > 0
        assert self.deal_id and self.intent_id and self.agent_id
    
    @property
    def chunk_count(self) -> int:
        """Get number of chunks in package"""
        return len(self.ordered_chunks)
    
    @property
    def document_types(self) -> List[DocumentRole]:
        """Get unique document types in package"""
        return list(set(chunk.document_role for chunk in self.ordered_chunks))
    
    @property
    def section_types(self) -> List[SectionType]:
        """Get unique section types in package"""
        return list(set(chunk.section_type for chunk in self.ordered_chunks))
    
    @property
    def has_exclusions(self) -> bool:
        """Check if there are any exclusions"""
        return len(self.exclusions_note.strip()) > 0

@dataclass(frozen=True)
class RetrievalSummary:
    """Summary of retrieval operation"""
    deal_id: str
    intent_id: str
    agent_id: str
    status: RetrievalStatus
    chunks_inspected: int
    chunks_selected: int
    tokens_selected: int
    risk_distribution: Dict[str, int]
    structural_coverage: Dict[str, bool]
    exclusions: List[str]
    confidence_score: float
    processing_time_ms: int
    
    @property
    def selection_ratio(self) -> float:
        """Get ratio of selected to inspected chunks"""
        if self.chunks_inspected == 0:
            return 0.0
        return self.chunks_selected / self.chunks_inspected
    
    @property
    def is_successful(self) -> bool:
        """Check if retrieval was successful"""
        return self.status == RetrievalStatus.SUCCESS
    
    @property
    def has_gaps(self) -> bool:
        """Check if there are structural gaps"""
        return not all(self.structural_coverage.values())

@dataclass(frozen=True)
class CoverageReport:
    """Report on structural coverage"""
    required_zones: List[str]
    covered_zones: List[str]
    partial_zones: List[str]
    missing_zones: List[str]
    coverage_percentage: float
    
    @property
    def is_complete(self) -> bool:
        """Check if coverage is complete"""
        return self.coverage_percentage >= 0.95
    
    @property
    def has_critical_gaps(self) -> bool:
        """Check if critical zones are missing"""
        return len(self.missing_zones) > 0

# ============================================================================
# ERROR MODELS
# ============================================================================

@dataclass(frozen=True)
class RetrievalError:
    """Error information for failed retrievals"""
    error_code: str
    error_message: str
    deal_id: str
    intent_id: str
    agent_id: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_retryable(self) -> bool:
        """Check if error is retryable"""
        retryable_codes = ["MEM0_UNAVAILABLE", "VECTOR_DB_TIMEOUT", "RATE_LIMITED"]
        return self.error_code in retryable_codes

class RetrievalErrorCode(str, Enum):
    """Standard error codes for retrieval operations"""
    NO_CHUNKS_FOUND = "NO_CHUNKS_FOUND"
    TOKEN_BUDGET_EXCEEDED = "TOKEN_BUDGET_EXCEEDED"
    MEM0_UNAVAILABLE = "MEM0_UNAVAILABLE"
    VECTOR_DB_ERROR = "VECTOR_DB_ERROR"
    INVALID_INTENT = "INVALID_INTENT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    STRUCTURAL_GAPS = "STRUCTURAL_GAPS"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"

# ============================================================================
# METRICS MODELS
# ============================================================================

@dataclass
class RetrievalMetrics:
    """Performance and quality metrics for retrieval operations"""
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
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def avg_chunk_latency_ms(self) -> float:
        """Average latency per chunk"""
        if self.chunks_inspected == 0:
            return 0.0
        return self.total_latency_ms / self.chunks_inspected
    
    @property
    def cost_per_token(self) -> float:
        """Cost per token selected"""
        if self.tokens_selected == 0:
            return 0.0
        return self.estimated_cost_usd / self.tokens_selected

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_deal_id(deal_id: str) -> bool:
    """Validate deal ID format"""
    return len(deal_id) > 0 and deal_id.replace("-", "").replace("_", "").isalnum()

def validate_chunk_id(chunk_id: str) -> bool:
    """Validate chunk ID is a valid hash"""
    return len(chunk_id) == 64 and all(c in "0123456789abcdef" for c in chunk_id.lower())

def validate_token_budget(budget: int, min_budget: int = 1000, max_budget: int = 8000) -> bool:
    """Validate token budget is within reasonable bounds"""
    return min_budget <= budget <= max_budget

def validate_ocr_confidence(confidence: float) -> bool:
    """Validate OCR confidence score"""
    return 0.0 <= confidence <= 1.0

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_retrieval_intent(
    deal_id: str,
    agent_id: str,
    query: str,
    sections: List[SectionType],
    risk_profile: RiskProfile = RiskProfile.BALANCED,
    max_tokens: int = 4000,
    required_zones: Optional[List[str]] = None
) -> RetrievalIntent:
    """Factory function to create retrieval intent with validation"""
    return RetrievalIntent(
        intent_id=f"{agent_id}_{risk_profile.value.lower()}",
        deal_id=deal_id,
        agent_id=agent_id,
        query_text=query,
        target_section_types=sections,
        risk_profile=risk_profile,
        max_tokens_budget=max_tokens,
        required_structural_zones=required_zones
    )

def create_context_package(
    deal_id: str,
    intent_id: str,
    agent_id: str,
    chunks: List[DocumentChunk],
    exclusions: List[str] = None
) -> ContextPackage:
    """Factory function to create context package from chunks"""
    # Generate structural TOC
    toc_sections = []
    for chunk in chunks:
        toc_sections.append(f"{chunk.section_type.value} ({chunk.document_role.value}, p{chunk.page_range})")
    
    structural_toc = " | ".join(toc_sections)
    exclusions_note = "; ".join(exclusions) if exclusions else "No structural exclusions"
    
    # Generate risk summary
    risk_keywords = ["caveat", "lien", "deficiency", "arrears", "penalty"]
    risk_count = sum(
        sum(1 for keyword in risk_keywords if keyword in chunk.content.lower())
        for chunk in chunks
    )
    risk_summary = f"{risk_count} risk indicators detected"
    
    return ContextPackage(
        deal_id=deal_id,
        intent_id=intent_id,
        agent_id=agent_id,
        ordered_chunks=[ChunkReference.from_chunk(chunk) for chunk in chunks],
        structural_toc=structural_toc,
        exclusions_note=exclusions_note,
        total_tokens=sum(chunk.token_count for chunk in chunks),
        risk_summary=risk_summary
    )

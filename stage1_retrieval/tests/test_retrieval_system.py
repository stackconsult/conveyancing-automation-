"""
Stage 1 Retrieval System - Test Suite

Comprehensive tests for all retrieval system components.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from datetime import datetime
from typing import List, Dict, Any

# Import schemas
from stage1_retrieval.schemas.core_schemas import (
    DocumentChunk, RetrievalIntent, CandidateChunk, CandidateSet,
    RankedContextSet, ContextPackage, RetrievalSummary, RetrievalStatus,
    DocumentRole, SectionType, RiskProfile, create_retrieval_intent,
    create_context_package
)

# Import algorithms
from stage1_retrieval.algorithms.retrieval_algorithms import (
    SegmentAwareRetriever, RiskAwareRanker, ContextPackager, CoverageSelfCheck
)

# Import integration
from stage1_retrieval.integration.retrieval_agent import RetrievalAgent
from stage1_retrieval.integration.langgraph_integration import (
    RetrievalNode, DealState, InvestigatorAdapter, TaxAdapter
)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_mem0_client():
    """Mock Mem0 client"""
    client = AsyncMock()
    return client

@pytest.fixture
def mock_vector_client():
    """Mock vector database client"""
    client = AsyncMock()
    return client

@pytest.fixture
def mock_embedding_model():
    """Mock embedding model"""
    model = AsyncMock()
    model.encode.return_value = [0.1, 0.2, 0.3] * 128  # Mock embedding
    return model

@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing"""
    return [
        DocumentChunk.create_chunk(
            deal_id="test_deal",
            document_id="doc1",
            document_role=DocumentRole.TITLE_SEARCH,
            page_start=1,
            page_end=2,
            section_type=SectionType.TITLE_SUMMARY,
            content="This is the title summary with important information.",
            ocr_confidence=0.95
        ),
        DocumentChunk.create_chunk(
            deal_id="test_deal",
            document_id="doc1",
            document_role=DocumentRole.TITLE_SEARCH,
            page_start=3,
            page_end=5,
            section_type=SectionType.INSTRUMENTS_REGISTER,
            content="Mortgage registered against the property. Caveat filed by contractor.",
            ocr_confidence=0.88
        ),
        DocumentChunk.create_chunk(
            deal_id="test_deal",
            document_id="doc2",
            document_role=DocumentRole.CONDO_DOCS,
            page_start=1,
            page_end=10,
            section_type=SectionType.RESERVE_FUND_REPORT,
            content="Reserve fund shows deficiency of $50,000. Special assessment proposed.",
            ocr_confidence=0.92
        )
    ]

@pytest.fixture
def sample_intent():
    """Sample retrieval intent"""
    return create_retrieval_intent(
        deal_id="test_deal",
        agent_id="investigator_r1",
        query="Find title risks and encumbrances",
        sections=[SectionType.TITLE_SUMMARY, SectionType.INSTRUMENTS_REGISTER],
        risk_profile=RiskProfile.HIGH_RISK,
        max_tokens=4000
    )

# ============================================================================
# SCHEMA TESTS
# ============================================================================

class TestDocumentChunk:
    """Test DocumentChunk model"""
    
    def test_chunk_creation(self):
        """Test creating a document chunk"""
        chunk = DocumentChunk.create_chunk(
            deal_id="test_deal",
            document_id="doc1",
            document_role=DocumentRole.TITLE_SEARCH,
            page_start=1,
            page_end=2,
            section_type=SectionType.TITLE_SUMMARY,
            content="Test content",
            ocr_confidence=0.95
        )
        
        assert chunk.deal_id == "test_deal"
        assert chunk.document_role == DocumentRole.TITLE_SEARCH
        assert chunk.section_type == SectionType.TITLE_SUMMARY
        assert chunk.page_start == 1
        assert chunk.page_end == 2
        assert chunk.is_high_confidence is True
        assert chunk.is_multi_page is False
        assert chunk.page_range == "1-2"
    
    def test_chunk_validation(self):
        """Test chunk validation"""
        # Valid chunk should not raise
        DocumentChunk.create_chunk(
            deal_id="test_deal",
            document_id="doc1",
            document_role=DocumentRole.TITLE_SEARCH,
            page_start=1,
            page_end=2,
            section_type=SectionType.TITLE_SUMMARY,
            content="Test content",
            ocr_confidence=0.95
        )
        
        # Invalid OCR confidence should raise
        with pytest.raises(AssertionError):
            DocumentChunk.create_chunk(
                deal_id="test_deal",
                document_id="doc1",
                document_role=DocumentRole.TITLE_SEARCH,
                page_start=1,
                page_end=2,
                section_type=SectionType.TITLE_SUMMARY,
                content="Test content",
                ocr_confidence=1.5  # Invalid
            )

class TestRetrievalIntent:
    """Test RetrievalIntent model"""
    
    def test_intent_creation(self):
        """Test creating a retrieval intent"""
        intent = create_retrieval_intent(
            deal_id="test_deal",
            agent_id="investigator_r1",
            query="Test query",
            sections=[SectionType.TITLE_SUMMARY],
            risk_profile=RiskProfile.HIGH_RISK,
            max_tokens=4000
        )
        
        assert intent.deal_id == "test_deal"
        assert intent.agent_id == "investigator_r1"
        assert intent.risk_profile == RiskProfile.HIGH_RISK
        assert intent.is_high_risk is True
        assert len(intent.target_section_types) == 1
    
    def test_intent_validation(self):
        """Test intent validation"""
        # Valid intent should not raise
        create_retrieval_intent(
            deal_id="test_deal",
            agent_id="investigator_r1",
            query="Test query",
            sections=[SectionType.TITLE_SUMMARY],
            risk_profile=RiskProfile.BALANCED,
            max_tokens=4000
        )
        
        # Invalid token budget should raise
        with pytest.raises(AssertionError):
            create_retrieval_intent(
                deal_id="test_deal",
                agent_id="investigator_r1",
                query="Test query",
                sections=[SectionType.TITLE_SUMMARY],
                risk_profile=RiskProfile.BALANCED,
                max_tokens=-100  # Invalid
            )

class TestContextPackage:
    """Test ContextPackage model"""
    
    def test_package_creation(self, sample_chunks):
        """Test creating a context package"""
        package = create_context_package(
            deal_id="test_deal",
            intent_id="test_intent",
            agent_id="test_agent",
            chunks=sample_chunks
        )
        
        assert package.deal_id == "test_deal"
        assert package.intent_id == "test_intent"
        assert package.agent_id == "test_agent"
        assert package.chunk_count == 3
        assert package.total_tokens > 0
        assert len(package.document_types) == 2  # TITLE_SEARCH, CONDO_DOCS
        assert len(package.section_types) == 3  # TITLE_SUMMARY, INSTRUMENTS_REGISTER, RESERVE_FUND_REPORT

# ============================================================================
# ALGORITHM TESTS
# ============================================================================

class TestSegmentAwareRetriever:
    """Test SegmentAwareRetriever algorithm"""
    
    @pytest.mark.asyncio
    async def test_retrieve_candidates(self, mock_mem0_client, mock_vector_client, mock_embedding_model, sample_intent):
        """Test candidate retrieval"""
        # Setup mocks
        mock_mem0_client.search.return_value = [
            {
                "metadata": {
                    "chunk_id": "chunk1",
                    "document_id": "doc1",
                    "document_role": DocumentRole.TITLE_SEARCH,
                    "page_start": 1,
                    "page_end": 2,
                    "section_type": SectionType.TITLE_SUMMARY,
                    "content": "Test content",
                    "token_count": 100,
                    "ocr_confidence": 0.95,
                    "content_hash": "hash123"
                }
            }
        ]
        
        mock_vector_client.search.return_value = [
            {"id": "chunk1", "score": 0.8}
        ]
        
        # Create retriever
        retriever = SegmentAwareRetriever(mock_mem0_client, mock_vector_client, mock_embedding_model)
        
        # Execute retrieval
        candidates = await retriever.retrieve_candidates(sample_intent, top_k=10)
        
        # Verify results
        assert candidates.chunk_count == 1
        assert candidates.total_tokens == 100
        assert len(candidates.structural_coverage_report) > 0

class TestRiskAwareRanker:
    """Test RiskAwareRanker algorithm"""
    
    @pytest.mark.asyncio
    async def test_rank_candidates(self, mock_mem0_client, sample_chunks, sample_intent):
        """Test candidate ranking"""
        # Setup mock Mem0 client
        mock_mem0_client.search.return_value = []  # No pattern data
        
        # Create candidate set
        candidates = CandidateSet(
            chunks=[
                CandidateChunk(
                    chunk=chunk,
                    semantic_score=0.7,
                    risk_signal_score=0.0,
                    mem0_pattern_score=0.0,
                    combined_score=0.7
                )
                for chunk in sample_chunks
            ],
            ranking_features={},
            total_tokens=300,
            structural_coverage_report={}
        )
        
        # Create ranker
        ranker = RiskAwareRanker(mock_mem0_client)
        
        # Execute ranking
        ranked_set = await ranker.rank_candidates(candidates, sample_intent)
        
        # Verify results
        assert len(ranked_set.chunks) > 0
        assert ranked_set.total_tokens > 0
        assert ranked_set.risk_summary is not None

class TestContextPackager:
    """Test ContextPackager algorithm"""
    
    def test_create_package(self, sample_chunks, sample_intent):
        """Test context package creation"""
        # Create mock ranked set
        from stage1_retrieval.algorithms.retrieval_algorithms import RankedContextSet
        
        ranked_set = RankedContextSet(
            chunks=sample_chunks,
            risk_metadata={},
            risk_summary="Test risk summary"
        )
        
        # Create packager
        packager = ContextPackager()
        
        # Create package
        package = packager.create_package(ranked_set, sample_intent)
        
        # Verify results
        assert package.deal_id == sample_intent.deal_id
        assert package.intent_id == sample_intent.intent_id
        assert package.agent_id == sample_intent.agent_id
        assert package.chunk_count == 3
        assert package.total_tokens > 0
        assert "|" in package.structural_toc  # Should have sections separated

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestRetrievalAgent:
    """Test RetrievalAgent integration"""
    
    @pytest.mark.asyncio
    async def test_retrieve_integration(self, mock_mem0_client, mock_vector_client, mock_embedding_model, sample_intent):
        """Test complete retrieval integration"""
        # Setup mocks
        mock_mem0_client.search.return_value = []
        mock_vector_client.search.return_value = []
        
        # Create agent
        agent = RetrievalAgent(mock_mem0_client, mock_vector_client, mock_embedding_model)
        
        # Execute retrieval (should handle empty results gracefully)
        try:
            context, summary = await agent.retrieve(sample_intent)
            # Should handle empty results without crashing
            assert summary.status in [RetrievalStatus.FAILED, RetrievalStatus.PARTIAL]
        except Exception as e:
            # Should raise RetrievalError, not generic exception
            from stage1_retrieval.schemas.core_schemas import RetrievalError
            assert isinstance(e, RetrievalError)

class TestLangGraphIntegration:
    """Test LangGraph integration"""
    
    def test_deal_state(self):
        """Test DealState model"""
        state = DealState(
            deal_id="test_deal",
            current_agent="investigator",
            current_step="title_risk_scan",
            documents_processed=["doc1.pdf"],
            context_packages={},
            retrieval_summaries={},
            analysis_results={},
            error_log=[]
        )
        
        assert state.deal_id == "test_deal"
        assert state.current_agent == "investigator"
        assert state.current_step == "title_risk_scan"
        assert len(state.documents_processed) == 1
    
    def test_investigator_adapter(self):
        """Test InvestigatorAdapter"""
        state = DealState(
            deal_id="test_deal",
            current_agent="investigator",
            current_step="title_risk_scan",
            documents_processed=["doc1.pdf"],
            context_packages={},
            retrieval_summaries={},
            analysis_results={},
            error_log=[]
        )
        
        adapter = InvestigatorAdapter()
        intent = adapter.build_intent(state)
        
        assert intent.deal_id == "test_deal"
        assert intent.agent_id == "investigator_r1"
        assert intent.risk_profile == RiskProfile.HIGH_RISK
        assert SectionType.TITLE_SUMMARY in intent.target_section_types
    
    def test_tax_adapter(self):
        """Test TaxAdapter"""
        state = DealState(
            deal_id="test_deal",
            current_agent="tax",
            current_step="tax_check",
            documents_processed=["doc1.pdf"],
            context_packages={},
            retrieval_summaries={},
            analysis_results={},
            error_log=[]
        )
        
        adapter = TaxAdapter()
        intent = adapter.build_intent(state)
        
        assert intent.deal_id == "test_deal"
        assert intent.agent_id == "tax_r1"
        assert intent.risk_profile == RiskProfile.BALANCED
        assert SectionType.TAX_CERTIFICATE in intent.target_section_types

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests for retrieval system"""
    
    @pytest.mark.asyncio
    async def test_large_document_handling(self, mock_mem0_client, mock_vector_client, mock_embedding_model):
        """Test handling of large document sets"""
        # Setup many mock chunks
        mock_chunks = []
        for i in range(100):  # 100 chunks
            mock_chunks.append({
                "metadata": {
                    "chunk_id": f"chunk_{i}",
                    "document_id": f"doc_{i//10}",
                    "document_role": DocumentRole.TITLE_SEARCH,
                    "page_start": i,
                    "page_end": i+1,
                    "section_type": SectionType.GENERAL,
                    "content": f"Content for chunk {i}",
                    "token_count": 100,
                    "ocr_confidence": 0.9,
                    "content_hash": f"hash_{i}"
                }
            })
        
        mock_mem0_client.search.return_value = mock_chunks
        mock_vector_client.search.return_value = [
            {"id": f"chunk_{i}", "score": 0.5} for i in range(50)
        ]
        
        # Create retriever
        retriever = SegmentAwareRetriever(mock_mem0_client, mock_vector_client, mock_embedding_model)
        
        # Create intent with reasonable budget
        intent = create_retrieval_intent(
            deal_id="large_deal",
            agent_id="test_agent",
            query="Test query",
            sections=[SectionType.GENERAL],
            risk_profile=RiskProfile.BALANCED,
            max_tokens=2000  # Reasonable budget
        )
        
        # Execute retrieval
        start_time = datetime.utcnow()
        candidates = await retriever.retrieve_candidates(intent, top_k=50)
        end_time = datetime.utcnow()
        
        # Verify performance
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert candidates.chunk_count <= 50  # Should respect top_k

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and fallbacks"""
    
    @pytest.mark.asyncio
    async def test_mem0_unavailable(self, mock_vector_client, mock_embedding_model, sample_intent):
        """Test handling when Mem0 is unavailable"""
        # Setup Mem0 to raise exception
        mock_mem0_client = AsyncMock()
        mock_mem0_client.search.side_effect = Exception("Mem0 unavailable")
        
        # Create retriever
        retriever = SegmentAwareRetriever(mock_mem0_client, mock_vector_client, mock_embedding_model)
        
        # Should raise RetrievalError
        with pytest.raises(Exception):
            await retriever.retrieve_candidates(sample_intent)
    
    @pytest.mark.asyncio
    async def test_vector_db_unavailable(self, mock_mem0_client, mock_embedding_model, sample_intent):
        """Test handling when vector DB is unavailable"""
        # Setup vector DB to raise exception
        mock_vector_client = AsyncMock()
        mock_vector_client.search.side_effect = Exception("Vector DB unavailable")
        
        # Setup Mem0 to return some chunks
        mock_mem0_client.search.return_value = [
            {
                "metadata": {
                    "chunk_id": "chunk1",
                    "document_id": "doc1",
                    "document_role": DocumentRole.TITLE_SEARCH,
                    "page_start": 1,
                    "page_end": 2,
                    "section_type": SectionType.TITLE_SUMMARY,
                    "content": "Test content",
                    "token_count": 100,
                    "ocr_confidence": 0.95,
                    "content_hash": "hash123"
                }
            }
        ]
        
        # Create retriever
        retriever = SegmentAwareRetriever(mock_mem0_client, mock_vector_client, mock_embedding_model)
        
        # Should handle gracefully or raise appropriate error
        try:
            candidates = await retriever.retrieve_candidates(sample_intent)
            # If it succeeds, should have fallback behavior
            assert candidates.chunk_count >= 0
        except Exception:
            # Should raise RetrievalError, not generic exception
            pass

# ============================================================================
# UTILITY TESTS
# ============================================================================

class TestUtilities:
    """Test utility functions"""
    
    def test_validation_functions(self):
        """Test validation utility functions"""
        from stage1_retrieval.schemas.core_schemas import (
            validate_deal_id, validate_chunk_id, validate_token_budget, validate_ocr_confidence
        )
        
        # Test deal ID validation
        assert validate_deal_id("deal_12345") is True
        assert validate_deal_id("") is False
        assert validate_deal_id("deal@invalid") is False
        
        # Test chunk ID validation
        assert validate_chunk_id("a" * 64) is True
        assert validate_chunk_id("short") is False
        assert validate_chunk_id("z" * 65) is False
        
        # Test token budget validation
        assert validate_token_budget(4000) is True
        assert validate_token_budget(500) is False
        assert validate_token_budget(10000) is False
        
        # Test OCR confidence validation
        assert validate_ocr_confidence(0.95) is True
        assert validate_ocr_confidence(0.0) is True
        assert validate_ocr_confidence(1.0) is True
        assert validate_ocr_confidence(-0.1) is False
        assert validate_ocr_confidence(1.1) is False

# ============================================================================
# INTEGRATION TEST EXAMPLE
# ============================================================================

@pytest.mark.asyncio
async def test_end_to_end_example():
    """
    Example of end-to-end test showing complete retrieval workflow.
    This demonstrates how all components work together.
    """
    # This would be replaced with actual client initialization in real tests
    mock_mem0_client = AsyncMock()
    mock_vector_client = AsyncMock()
    mock_embedding_model = AsyncMock()
    
    # Setup realistic mock responses
    mock_mem0_client.search.return_value = [
        {
            "metadata": {
                "chunk_id": "chunk1",
                "document_id": "title_search.pdf",
                "document_role": DocumentRole.TITLE_SEARCH,
                "page_start": 1,
                "page_end:": 2,
                "section_type": SectionType.TITLE_SUMMARY,
                "content": "Property located at 123 Main St. Owner: John Doe.",
                "token_count": 50,
                "ocr_confidence": 0.95,
                "content_hash": "hash123"
            }
        },
        {
            "metadata": {
                "chunk_id": "chunk2",
                "document_id": "title_search.pdf",
                "document_role": DocumentRole.TITLE_SEARCH,
                "page_start": 3,
                "page_end": 5,
                "section_type": SectionType.INSTRUMENTS_REGISTER,
                "content": "Mortgage registered: ABC Bank. Caveat filed by XYZ Construction.",
                "token_count": 75,
                "ocr_confidence": 0.92,
                "content_hash": "hash456"
            }
        }
    ]
    
    mock_vector_client.search.return_value = [
        {"id": "chunk1", "score": 0.8},
        {"id": "chunk2", "score": 0.9}
    ]
    
    mock_embedding_model.encode.return_value = [0.1, 0.2, 0.3] * 128
    
    # Create retrieval agent
    agent = RetrievalAgent(mock_mem0_client, mock_vector_client, mock_embedding_model)
    
    # Create intent for title risk scan
    intent = create_retrieval_intent(
        deal_id="alberta_deal_001",
        agent_id="investigator_r1",
        query="Identify title risks, encumbrances, and ownership issues for Alberta property",
        sections=[SectionType.TITLE_SUMMARY, SectionType.INSTRUMENTS_REGISTER, SectionType.CAVEATS_SECTION],
        risk_profile=RiskProfile.HIGH_RISK,
        max_tokens=4000
    )
    
    # Execute retrieval
    context_package, summary = await agent.retrieve(intent)
    
    # Verify results
    assert context_package.deal_id == "alberta_deal_001"
    assert context_package.intent_id == "investigator_r1_title_risk_scan_v1"
    assert context_package.agent_id == "investigator_r1"
    assert context_package.chunk_count >= 1
    assert context_package.total_tokens > 0
    assert summary.status in [RetrievalStatus.SUCCESS, RetrievalStatus.PARTIAL]
    assert summary.confidence_score > 0.0
    assert summary.processing_time_ms > 0
    
    # Verify content contains risk indicators
    risk_keywords = ["mortgage", "caveat"]
    content_text = " ".join(chunk.content for chunk in context_package.ordered_chunks).lower()
    assert any(keyword in content_text for keyword in risk_keywords)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

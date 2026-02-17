"""
Stage 1 Retrieval System - Main RetrievalAgent

This module contains the main RetrievalAgent that orchestrates
the entire Stage 1 retrieval pipeline.
"""

import asyncio
from typing import Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from ..schemas.core_schemas import (
    RetrievalIntent, ContextPackage, RetrievalSummary, RetrievalStatus,
    RetrievalError, RetrievalErrorCode, RetrievalMetrics
)
from ..algorithms.retrieval_algorithms import (
    SegmentAwareRetriever, RiskAwareRanker, ContextPackager, CoverageSelfCheck
)

# ============================================================================
# MAIN RETRIEVAL AGENT
# ============================================================================

class RetrievalAgent:
    """
    Main orchestrator for Stage 1 Intelligent Retrieval.
    
    This agent bridges the gap between pre-processed documents
    and DeepSeek-R1 reasoning by providing the right slices
    of 100-page+ Alberta conveyancing files.
    """
    
    def __init__(
        self,
        mem0_client: Any,
        vector_client: Any,
        embedding_model: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the RetrievalAgent with required components.
        
        Args:
            mem0_client: Mem0 client for memory operations
            vector_client: Vector database client for semantic search
            embedding_model: Embedding model for semantic similarity
            config: Optional configuration overrides
        """
        self.config = config or {}
        
        # Initialize core components
        self.retriever = SegmentAwareRetriever(mem0_client, vector_client, embedding_model)
        self.ranker = RiskAwareRanker(mem0_client)
        self.packager = ContextPackager()
        self.coverage_checker = CoverageSelfCheck(self.retriever, self.ranker)
        
        # Performance tracking
        self.metrics_collector = MetricsCollector()
    
    async def retrieve(
        self,
        intent: RetrievalIntent
    ) -> Tuple[ContextPackage, RetrievalSummary]:
        """
        Execute the complete retrieval pipeline.
        
        Args:
            intent: Retrieval intent specifying what to find
            
        Returns:
            Tuple of (context_package, retrieval_summary)
            
        Raises:
            RetrievalError: If retrieval fails completely
        """
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Retrieve candidates
            candidates = await self._step_retrieve_candidates(intent)
            
            # Step 2: Rank candidates
            ranked_set = await self._step_rank_candidates(candidates, intent)
            
            # Step 3: Create initial context package
            initial_package = await self._step_create_package(ranked_set, intent)
            
            # Step 4: Validate coverage and patch gaps
            final_package, patches = await self._step_validate_coverage(
                initial_package, intent
            )
            
            # Step 5: Generate summary
            summary = self._generate_summary(
                intent, final_package, patches, start_time
            )
            
            # Step 6: Log metrics
            await self._log_metrics(intent, summary)
            
            return final_package, summary
            
        except Exception as e:
            # Convert to RetrievalError if needed
            if not isinstance(e, RetrievalError):
                error = RetrievalError(
                    error_code=RetrievalErrorCode.TIMEOUT,
                    error_message=f"Retrieval failed: {str(e)}",
                    deal_id=intent.deal_id,
                    intent_id=intent.intent_id,
                    agent_id=intent.agent_id,
                    timestamp=datetime.utcnow()
                )
            else:
                error = e
            
            # Generate error summary
            summary = RetrievalSummary(
                deal_id=intent.deal_id,
                intent_id=intent.intent_id,
                agent_id=intent.agent_id,
                status=RetrievalStatus.FAILED,
                chunks_inspected=0,
                chunks_selected=0,
                tokens_selected=0,
                risk_distribution={},
                structural_coverage={},
                exclusions=[],
                confidence_score=0.0,
                processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )
            
            raise error from e
    
    async def _step_retrieve_candidates(self, intent: RetrievalIntent):
        """Step 1: Retrieve candidate chunks"""
        with self.metrics_collector.time("retrieval"):
            return await self.retriever.retrieve_candidates(intent)
    
    async def _step_rank_candidates(
        self,
        candidates,
        intent: RetrievalIntent
    ):
        """Step 2: Rank candidates with risk awareness"""
        with self.metrics_collector.time("ranking"):
            return await self.ranker.rank_candidates(candidates, intent)
    
    async def _step_create_package(
        self,
        ranked_set,
        intent: RetrievalIntent
    ) -> ContextPackage:
        """Step 3: Create context package"""
        with self.metrics_collector.time("packaging"):
            return self.packager.create_package(ranked_set, intent)
    
    async def _step_validate_coverage(
        self,
        context_package: ContextPackage,
        intent: RetrievalIntent
    ) -> Tuple[ContextPackage, list]:
        """Step 4: Validate coverage and patch gaps"""
        with self.metrics_collector.time("coverage_check"):
            return await self.coverage_checker.validate_and_patch(
                context_package, intent
            )
    
    def _generate_summary(
        self,
        intent: RetrievalIntent,
        context_package: ContextPackage,
        patches: list,
        start_time: datetime
    ) -> RetrievalSummary:
        """Generate retrieval operation summary"""
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Determine status
        if context_package.total_tokens == 0:
            status = RetrievalStatus.FAILED
        elif patches:
            status = RetrievalStatus.PARTIAL
        else:
            status = RetrievalStatus.SUCCESS
        
        # Calculate confidence based on coverage and OCR confidence
        avg_ocr_confidence = sum(
            chunk.ocr_confidence for chunk in context_package.ordered_chunks
        ) / len(context_package.ordered_chunks) if context_package.ordered_chunks else 0.0
        
        confidence_score = min(avg_ocr_confidence, 0.95)  # Cap at 95%
        
        # Generate structural coverage
        structural_coverage = {}
        for section_type in intent.target_section_types:
            structural_coverage[section_type.value] = any(
                chunk.section_type == section_type 
                for chunk in context_package.ordered_chunks
            )
        
        return RetrievalSummary(
            deal_id=intent.deal_id,
            intent_id=intent.intent_id,
            agent_id=intent.agent_id,
            status=status,
            chunks_inspected=len(context_package.ordered_chunks) + len(patches),
            chunks_selected=len(context_package.ordered_chunks),
            tokens_selected=context_package.total_tokens,
            risk_distribution={},  # Would be populated from ranked_set
            structural_coverage=structural_coverage,
            exclusions=[context_package.exclusions_note] if context_package.has_exclusions else [],
            confidence_score=confidence_score,
            processing_time_ms=processing_time
        )
    
    async def _log_metrics(self, intent: RetrievalIntent, summary: RetrievalSummary):
        """Log performance and quality metrics"""
        metrics = RetrievalMetrics(
            deal_id=intent.deal_id,
            intent_id=intent.intent_id,
            agent_id=intent.agent_id,
            total_latency_ms=summary.processing_time_ms,
            mem0_query_time_ms=self.metrics_collector.get_time("retrieval"),
            vector_search_time_ms=0,  # Would be tracked in retriever
            ranking_time_ms=self.metrics_collector.get_time("ranking"),
            packaging_time_ms=self.metrics_collector.get_time("packaging"),
            chunks_inspected=summary.chunks_inspected,
            chunks_selected=summary.chunks_selected,
            tokens_selected=summary.tokens_selected,
            risk_distribution=summary.risk_distribution,
            structural_coverage_score=summary.confidence_score,
            embedding_api_calls=1,  # Simplified
            mem0_api_calls=2,  # Simplified
            estimated_cost_usd=0.01  # Simplified
        )
        
        # In production, send to monitoring system
        await self._send_metrics(metrics)
    
    async def _send_metrics(self, metrics: RetrievalMetrics):
        """Send metrics to monitoring system"""
        # Implementation depends on monitoring stack
        # Could be Prometheus, CloudWatch, etc.
        pass

# ============================================================================
# METRICS COLLECTOR
# ============================================================================

@dataclass
class TimingContext:
    """Context for timing operations"""
    name: str
    start_time: datetime
    
    def elapsed_ms(self) -> int:
        return int((datetime.utcnow() - self.start_time).total_seconds() * 1000)

class MetricsCollector:
    """Collects performance metrics during retrieval operations"""
    
    def __init__(self):
        self.timings = {}
        self.counters = {}
    
    def time(self, operation_name: str):
        """Context manager for timing operations"""
        return TimingContext(operation_name, datetime.utcnow())
    
    def get_time(self, operation_name: str) -> int:
        """Get elapsed time for an operation"""
        if operation_name in self.timings:
            return self.timings[operation_name].elapsed_ms()
        return 0
    
    def increment(self, counter_name: str, value: int = 1):
        """Increment a counter"""
        self.counters[counter_name] = self.counters.get(counter_name, 0) + value
    
    def get_counter(self, counter_name: str) -> int:
        """Get counter value"""
        return self.counters.get(counter_name, 0)
    
    def reset(self):
        """Reset all metrics"""
        self.timings.clear()
        self.counters.clear()

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_retrieval_agent(
    mem0_client: Any,
    vector_client: Any,
    embedding_model: Any,
    config: Optional[Dict[str, Any]] = None
) -> RetrievalAgent:
    """
    Factory function to create a configured RetrievalAgent.
    
    Args:
        mem0_client: Mem0 client instance
        vector_client: Vector database client instance
        embedding_model: Embedding model instance
        config: Optional configuration
        
    Returns:
        Configured RetrievalAgent instance
    """
    return RetrievalAgent(
        mem0_client=mem0_client,
        vector_client=vector_client,
        embedding_model=embedding_model,
        config=config
    )

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def retrieve_context(
    retrieval_agent: RetrievalAgent,
    deal_id: str,
    agent_id: str,
    query: str,
    sections: list,
    risk_profile: str = "BALANCED",
    max_tokens: int = 4000
) -> Tuple[ContextPackage, RetrievalSummary]:
    """
    Convenience function for common retrieval operations.
    
    Args:
        retrieval_agent: Configured RetrievalAgent
        deal_id: Deal identifier
        agent_id: Agent identifier
        query: Natural language query
        sections: List of section types to target
        risk_profile: Risk tolerance level
        max_tokens: Maximum token budget
        
    Returns:
        Tuple of (context_package, retrieval_summary)
    """
    from ..schemas.core_schemas import (
        RetrievalIntent, SectionType, RiskProfile, create_retrieval_intent
    )
    
    # Convert string inputs to enums
    section_types = [SectionType(st) for st in sections]
    risk_enum = RiskProfile(risk_profile)
    
    # Create intent
    intent = create_retrieval_intent(
        deal_id=deal_id,
        agent_id=agent_id,
        query=query,
        sections=section_types,
        risk_profile=risk_enum,
        max_tokens=max_tokens
    )
    
    # Execute retrieval
    return await retrieval_agent.retrieve(intent)

"""
Stage 1 Retrieval System - Algorithm Implementations

This module contains the core retrieval algorithms that implement
the intelligent document slicing functionality.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..schemas.core_schemas import (
    DocumentChunk, RetrievalIntent, CandidateChunk, CandidateSet,
    RankedContextSet, ContextPackage, RetrievalSummary, RetrievalStatus,
    RiskProfile, SectionType, RetrievalError, RetrievalErrorCode
)

# ============================================================================
# SEGMENT AWARE RETRIEVER
# ============================================================================

class SegmentAwareRetriever:
    """
    Retrieves structurally coherent candidates using hybrid search.
    Combines semantic similarity with structural filtering.
    """
    
    def __init__(
        self,
        mem0_client: Any,
        vector_client: Any,
        embedding_model: Any
    ):
        self.mem0_client = mem0_client
        self.vector_client = vector_client
        self.embedding_model = embedding_model
    
    async def retrieve_candidates(
        self,
        intent: RetrievalIntent,
        top_k: int = 50
    ) -> CandidateSet:
        """
        Retrieve candidates using hybrid semantic + structural search.
        
        Args:
            intent: Retrieval intent specifying what to find
            top_k: Maximum number of candidates to return
            
        Returns:
            CandidateSet with retrieved chunks and metadata
        """
        # Step 1: Structural filtering via Mem0
        filtered_chunks = await self._filter_by_structure(intent)
        
        if not filtered_chunks:
            return CandidateSet(
                chunks=[],
                ranking_features={"filtered_count": 0},
                total_tokens=0,
                structural_coverage_report={}
            )
        
        # Step 2: Semantic search over filtered chunks
        semantic_candidates = await self._semantic_search(intent, filtered_chunks, top_k)
        
        # Step 3: Enforce local narrative coherence
        coherent_chunks = self._group_by_coherence(semantic_candidates)
        
        # Step 4: Create candidate set with metadata
        return self._create_candidate_set(coherent_chunks, intent)
    
    async def _filter_by_structure(self, intent: RetrievalIntent) -> List[DocumentChunk]:
        """Filter chunks by deal scope and section types using Mem0"""
        try:
            # Build Mem0 filters
            filters = {
                "AND": [
                    {"user_id": f"conv-{intent.deal_id}"},
                    {"section_type": {"in": [st.value for st in intent.target_section_types]}}
                ]
            }
            
            # Query Mem0 for chunk metadata
            results = await self.mem0_client.search(
                query="",
                version="v2",
                filters=filters
            )
            
            # Convert results to DocumentChunk objects
            chunks = []
            for result in results:
                # Extract chunk metadata from Mem0 result
                metadata = result.get("metadata", {})
                chunk = DocumentChunk(
                    chunk_id=metadata.get("chunk_id"),
                    deal_id=intent.deal_id,
                    document_id=metadata.get("document_id"),
                    document_role=metadata.get("document_role"),
                    page_start=metadata.get("page_start"),
                    page_end=metadata.get("page_end"),
                    section_type=metadata.get("section_type"),
                    content=metadata.get("content", ""),
                    token_count=metadata.get("token_count", 0),
                    ocr_confidence=metadata.get("ocr_confidence", 0.0),
                    content_hash=metadata.get("content_hash"),
                    metadata=metadata
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            raise RetrievalError(
                error_code=RetrievalErrorCode.MEM0_UNAVAILABLE,
                error_message=f"Mem0 structural filtering failed: {str(e)}",
                deal_id=intent.deal_id,
                intent_id=intent.intent_id,
                agent_id=intent.agent_id,
                timestamp=datetime.utcnow()
            )
    
    async def _semantic_search(
        self,
        intent: RetrievalIntent,
        filtered_chunks: List[DocumentChunk],
        top_k: int
    ) -> List[Tuple[DocumentChunk, float]]:
        """Perform semantic search over filtered chunks"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_model.encode(intent.query_text)
            
            # Get chunk IDs for vector search
            chunk_ids = [chunk.chunk_id for chunk in filtered_chunks]
            
            # Search vector database
            vector_results = await self.vector_client.search(
                query_vector=query_embedding,
                filter={"chunk_id": {"in": chunk_ids}},
                limit=top_k
            )
            
            # Map results back to chunks with scores
            chunk_score_map = {chunk.chunk_id: chunk for chunk in filtered_chunks}
            scored_chunks = []
            
            for result in vector_results:
                chunk_id = result.get("id")
                score = result.get("score", 0.0)
                
                if chunk_id in chunk_score_map:
                    chunk = chunk_score_map[chunk_id]
                    scored_chunks.append((chunk, score))
            
            return scored_chunks
            
        except Exception as e:
            raise RetrievalError(
                error_code=RetrievalErrorCode.VECTOR_DB_ERROR,
                error_message=f"Vector search failed: {str(e)}",
                deal_id=intent.deal_id,
                intent_id=intent.intent_id,
                agent_id=intent.agent_id,
                timestamp=datetime.utcnow()
            )
    
    def _group_by_coherence(
        self,
        scored_chunks: List[Tuple[DocumentChunk, float]]
    ) -> List[Tuple[DocumentChunk, float]]:
        """Group adjacent page ranges for narrative coherence"""
        if not scored_chunks:
            return []
        
        # Sort by page range first, then by score
        scored_chunks.sort(key=lambda x: (x[0].page_start, -x[1]))
        
        # Group adjacent chunks
        coherent_groups = []
        current_group = [scored_chunks[0]]
        
        for chunk, score in scored_chunks[1:]:
            last_chunk, _ = current_group[-1]
            
            # Check if pages are adjacent (allow small gaps)
            if chunk.page_start <= last_chunk.page_end + 2:
                current_group.append((chunk, score))
            else:
                coherent_groups.append(current_group)
                current_group = [(chunk, score)]
        
        coherent_groups.append(current_group)
        
        # Flatten groups while preserving coherence
        coherent_chunks = []
        for group in coherent_groups:
            # Sort within group by score (descending)
            group.sort(key=lambda x: -x[1])
            coherent_chunks.extend(group)
        
        return coherent_chunks
    
    def _create_candidate_set(
        self,
        scored_chunks: List[Tuple[DocumentChunk, float]],
        intent: RetrievalIntent
    ) -> CandidateSet:
        """Create candidate set with metadata"""
        candidates = []
        total_tokens = 0
        
        for chunk, semantic_score in scored_chunks:
            candidate = CandidateChunk(
                chunk=chunk,
                semantic_score=semantic_score,
                risk_signal_score=0.0,  # Will be computed by ranker
                mem0_pattern_score=0.0,  # Will be computed by ranker
                combined_score=semantic_score  # Temporary
            )
            candidates.append(candidate)
            total_tokens += chunk.token_count
        
        # Generate structural coverage report
        structural_coverage = {}
        for section_type in intent.target_section_types:
            structural_coverage[section_type.value] = any(
                cand.chunk.section_type == section_type for cand in candidates
            )
        
        ranking_features = {
            "semantic_scores": [c.semantic_score for c in candidates],
            "section_distribution": {
                st.value: sum(1 for c in candidates if c.chunk.section_type == st)
                for st in SectionType
            },
            "ocr_confidence_distribution": [
                c.chunk.ocr_confidence for c in candidates
            ]
        }
        
        return CandidateSet(
            chunks=candidates,
            ranking_features=ranking_features,
            total_tokens=total_tokens,
            structural_coverage_report=structural_coverage
        )

# ============================================================================
# RISK AWARE RANKER
# ============================================================================

class RiskAwareRanker:
    """
    Re-ranks candidates using risk heuristics and historical patterns.
    Implements risk-weighted scoring without precision/recall tradeoff.
    """
    
    def __init__(self, mem0_client: Any):
        self.mem0_client = mem0_client
        
        # Risk signal keywords by section type
        self.risk_keywords = {
            SectionType.TITLE_SUMMARY: ["caveat", "lien", "encumbrance", "restriction"],
            SectionType.INSTRUMENTS_REGISTER: ["mortgage", "charge", "easement", "caveat"],
            SectionType.CAVEATS_SECTION: ["caveat", "lien", "judgment", "court"],
            SectionType.RESERVE_FUND_REPORT: ["deficiency", "shortfall", "underfunded"],
            SectionType.FINANCIAL_STATEMENTS: ["deficit", "loss", "liability"],
            SectionType.TAX_ARREARS: ["arrears", "penalty", "interest", "lien"],
            SectionType.SPECIAL_RESOLUTIONS: ["special assessment", "levy", "charge"]
        }
        
        # Risk profile weights
        self.risk_weights = {
            RiskProfile.HIGH_RISK: {"semantic": 0.3, "risk": 0.5, "pattern": 0.2},
            RiskProfile.BALANCED: {"semantic": 0.5, "risk": 0.3, "pattern": 0.2},
            RiskProfile.LOW_RISK: {"semantic": 0.7, "risk": 0.2, "pattern": 0.1}
        }
    
    async def rank_candidates(
        self,
        candidates: CandidateSet,
        intent: RetrievalIntent
    ) -> RankedContextSet:
        """
        Re-rank candidates using risk-weighted scoring.
        
        Args:
            candidates: Set of candidate chunks
            intent: Retrieval intent with risk profile
            
        Returns:
            RankedContextSet with re-ordered chunks
        """
        # Step 1: Compute risk signal scores
        await self._compute_risk_signals(candidates)
        
        # Step 2: Compute Mem0 pattern scores
        await self._compute_pattern_scores(candidates, intent)
        
        # Step 3: Combine scores based on risk profile
        self._combine_scores(candidates, intent)
        
        # Step 4: Enforce token budget with risk bias
        selected_chunks = self._select_with_budget(candidates, intent)
        
        # Step 5: Create ranked context set
        return self._create_ranked_set(selected_chunks, intent)
    
    async def _compute_risk_signals(self, candidates: CandidateSet):
        """Compute risk signal scores for each candidate"""
        for candidate in candidates.chunks:
            chunk = candidate.chunk
            section_type = chunk.section_type
            
            # Get risk keywords for this section type
            keywords = self.risk_keywords.get(section_type, [])
            
            # Count risk keywords in content
            content_lower = chunk.content.lower()
            risk_count = sum(1 for keyword in keywords if keyword in content_lower)
            
            # Normalize risk score (0-1)
            max_possible = len(keywords)
            if max_possible > 0:
                candidate.risk_signal_score = min(risk_count / max_possible, 1.0)
            else:
                candidate.risk_signal_score = 0.0
    
    async def _compute_pattern_scores(
        self,
        candidates: CandidateSet,
        intent: RetrievalIntent
    ):
        """Compute Mem0 pattern scores based on historical data"""
        for candidate in candidates.chunks:
            chunk = candidate.chunk
            
            # Extract entities from chunk for pattern lookup
            entities = self._extract_entities(chunk)
            
            # Search Mem0 for historical patterns
            pattern_score = 0.0
            if entities:
                try:
                    # Search for patterns related to entities
                    pattern_query = f"risk patterns {entities[0]}"  # Use first entity
                    results = await self.mem0_client.search(
                        query=pattern_query,
                        version="v2",
                        filters={"OR": [{"entity_id": entity} for entity in entities]}
                    )
                    
                    # Score based on number of matching patterns
                    pattern_score = min(len(results) / 5.0, 1.0)  # Normalize to 0-1
                    
                except Exception:
                    # If Mem0 search fails, assume no pattern data
                    pattern_score = 0.0
            
            candidate.mem0_pattern_score = pattern_score
    
    def _extract_entities(self, chunk: DocumentChunk) -> List[str]:
        """Extract potential entities (builder names, condo corps, etc.)"""
        # Simple entity extraction - in production, use NER
        entities = []
        
        # Look for common patterns
        content = chunk.content
        
        # Builder names (simplified)
        import re
        builder_pattern = r'\b([A-Z][a-z]+ (?:Homes|Construction|Builders|Developments))\b'
        matches = re.findall(builder_pattern, content)
        entities.extend(matches)
        
        # Condo corporation patterns
        condo_pattern = r'\b([A-Z]+ \d+ (?:Condo|Corporation|Corp))\b'
        matches = re.findall(condo_pattern, content)
        entities.extend(matches)
        
        return entities[:3]  # Limit to top 3 entities
    
    def _combine_scores(self, candidates: CandidateSet, intent: RetrievalIntent):
        """Combine semantic, risk, and pattern scores"""
        weights = self.risk_weights[intent.risk_profile]
        
        for candidate in candidates.chunks:
            combined = (
                weights["semantic"] * candidate.semantic_score +
                weights["risk"] * candidate.risk_signal_score +
                weights["pattern"] * candidate.mem0_pattern_score
            )
            candidate.combined_score = combined
    
    def _select_with_budget(
        self,
        candidates: CandidateSet,
        intent: RetrievalIntent
    ) -> List[DocumentChunk]:
        """Select chunks within token budget with risk bias"""
        # Sort by combined score (descending)
        sorted_candidates = sorted(
            candidates.chunks,
            key=lambda c: c.combined_score,
            reverse=True
        )
        
        selected_chunks = []
        used_tokens = 0
        structural_coverage = {st: False for st in intent.target_section_types}
        
        for candidate in sorted_candidates:
            chunk = candidate.chunk
            
            # Check token budget
            if used_tokens + chunk.token_count > intent.max_tokens_budget:
                continue
            
            # Prioritize structural coverage
            if not structural_coverage[chunk.section_type]:
                selected_chunks.append(chunk)
                used_tokens += chunk.token_count
                structural_coverage[chunk.section_type] = True
            else:
                # Add if we have budget and it's a high-risk chunk
                if candidate.risk_signal_score > 0.3:
                    selected_chunks.append(chunk)
                    used_tokens += chunk.token_count
        
        return selected_chunks
    
    def _create_ranked_set(
        self,
        chunks: List[DocumentChunk],
        intent: RetrievalIntent
    ) -> RankedContextSet:
        """Create ranked context set with metadata"""
        # Generate risk metadata
        risk_metadata = {
            "risk_distribution": self._compute_risk_distribution(chunks),
            "high_risk_chunks": [
                chunk.chunk_id for chunk in chunks
                if self._has_high_risk_keywords(chunk)
            ],
            "structural_coverage": {
                st.value: any(chunk.section_type == st for chunk in chunks)
                for st in intent.target_section_types
            }
        }
        
        # Generate risk summary
        high_risk_count = len(risk_metadata["high_risk_chunks"])
        risk_summary = f"{high_risk_count} high-risk chunks identified"
        
        return RankedContextSet(
            chunks=chunks,
            risk_metadata=risk_metadata,
            risk_summary=risk_summary
        )
    
    def _compute_risk_distribution(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Compute distribution of risk levels"""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for chunk in chunks:
            if self._has_high_risk_keywords(chunk):
                distribution["high"] += 1
            elif self._has_medium_risk_keywords(chunk):
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    def _has_high_risk_keywords(self, chunk: DocumentChunk) -> bool:
        """Check if chunk contains high-risk keywords"""
        high_risk_keywords = ["caveat", "lien", "judgment", "deficiency", "arrears"]
        content_lower = chunk.content.lower()
        return any(keyword in content_lower for keyword in high_risk_keywords)
    
    def _has_medium_risk_keywords(self, chunk: DocumentChunk) -> bool:
        """Check if chunk contains medium-risk keywords"""
        medium_risk_keywords = ["agreement", "terms", "conditions", "restriction", "encumbrance"]
        content_lower = chunk.content.lower()
        return any(keyword in content_lower for keyword in medium_risk_keywords)

# ============================================================================
# CONTEXT PACKAGER
# ============================================================================

class ContextPackager:
    """
    Assembles bounded context packages for downstream agents.
    Optimizes layout for DeepSeek-R1 with structural awareness.
    """
    
    def __init__(self):
        self.max_section_name_length = 50
    
    def create_package(
        self,
        ranked_set: RankedContextSet,
        intent: RetrievalIntent,
        exclusions: List[str] = None
    ) -> ContextPackage:
        """
        Create a context package from ranked chunks.
        
        Args:
            ranked_set: Ranked chunks and metadata
            intent: Original retrieval intent
            exclusions: List of excluded structural zones
            
        Returns:
            ContextPackage ready for downstream consumption
        """
        # Step 1: Order chunks by document and page (narrative coherence)
        ordered_chunks = self._order_by_narrative(ranked_set.chunks)
        
        # Step 2: Generate structural TOC
        structural_toc = self._generate_structural_toc(ordered_chunks)
        
        # Step 3: Create exclusions note
        exclusions_note = self._create_exclusions_note(exclusions)
        
        # Step 4: Create context package
        return ContextPackage(
            deal_id=intent.deal_id,
            intent_id=intent.intent_id,
            agent_id=intent.agent_id,
            ordered_chunks=ordered_chunks,
            structural_toc=structural_toc,
            exclusions_note=exclusions_note,
            total_tokens=sum(chunk.token_count for chunk in ranked_set.chunks),
            risk_summary=ranked_set.risk_summary
        )
    
    def _order_by_narrative(self, chunks: List[DocumentChunk]) -> List[Any]:
        """Order chunks by document and page for narrative coherence"""
        from ..schemas.core_schemas import ChunkReference
        
        # Sort by document role, then by page range
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (c.document_role.value, c.page_start, c.page_end)
        )
        
        # Convert to chunk references
        return [ChunkReference.from_chunk(chunk) for chunk in sorted_chunks]
    
    def _generate_structural_toc(self, chunks: List[Any]) -> str:
        """Generate structural table of contents"""
        from ..schemas.core_schemas import ChunkReference
        
        sections = []
        current_doc = None
        
        for chunk in chunks:
            if chunk.document_role != current_doc:
                # New document section
                current_doc = chunk.document_role
                sections.append(f"\n{chunk.document_role.value}:")
            
            # Add chunk section
            section_name = chunk.section_type.value
            if len(section_name) > self.max_section_name_length:
                section_name = section_name[:self.max_section_name_length-3] + "..."
            
            sections.append(f"  {section_name} (p{chunk.page_range})")
        
        return " | ".join(sections).strip()
    
    def _create_exclusions_note(self, exclusions: List[str]) -> str:
        """Create note about excluded structural zones"""
        if not exclusions:
            return "No structural exclusions"
        
        return f"Excluded zones: " + "; ".join(exclusions)

# ============================================================================
# COVERAGE SELF CHECK
# ============================================================================

class CoverageSelfCheck:
    """
    Validates structural coverage and patches obvious gaps.
    Ensures no critical zones are missed.
    """
    
    def __init__(self, retriever: SegmentAwareRetriever, ranker: RiskAwareRanker):
        self.retriever = retriever
        self.ranker = ranker
        
        # Required zones per intent type
        self.required_zones = {
            "title_risk_scan_v1": ["front_summary", "instruments_register", "caveats_section"],
            "condo_reserve_fund_health_v1": ["reserve_fund_report", "financial_statements", "special_resolutions"],
            "tax_arrears_check_v1": ["tax_certificate", "municipal_arrears", "tax_liens"]
        }
    
    async def validate_and_patch(
        self,
        context_package: ContextPackage,
        intent: RetrievalIntent
    ) -> Tuple[ContextPackage, List[str]]:
        """
        Validate structural coverage and patch gaps if needed.
        
        Args:
            context_package: Current context package
            intent: Original retrieval intent
            
        Returns:
            Tuple of (updated package, list of patches applied)
        """
        # Step 1: Check required zones
        required_zones = self.required_zones.get(intent.intent_id, [])
        coverage_report = self._check_coverage(context_package, required_zones)
        
        # Step 2: Identify missing zones
        missing_zones = [
            zone for zone, covered in coverage_report.items()
            if not covered
        ]
        
        patches_applied = []
        
        # Step 3: Patch missing zones
        if missing_zones:
            patched_package = await self._patch_missing_zones(
                context_package, intent, missing_zones
            )
            patches_applied = missing_zones
        else:
            patched_package = context_package
        
        return patched_package, patches_applied
    
    def _check_coverage(
        self,
        context_package: ContextPackage,
        required_zones: List[str]
    ) -> Dict[str, bool]:
        """Check which required zones are covered"""
        coverage = {}
        
        for zone in required_zones:
            # Check if zone is mentioned in structural TOC
            covered = any(zone.lower() in section.lower() 
                       for section in context_package.structural_toc.split("|"))
            coverage[zone] = covered
        
        return coverage
    
    async def _patch_missing_zones(
        self,
        context_package: ContextPackage,
        intent: RetrievalIntent,
        missing_zones: List[str]
    ) -> ContextPackage:
        """Patch missing zones with targeted retrieval"""
        # Create focused intent for missing zones
        focused_sections = self._map_zones_to_sections(missing_zones)
        
        focused_intent = RetrievalIntent(
            intent_id=f"{intent.intent_id}_patch",
            deal_id=intent.deal_id,
            agent_id=intent.agent_id,
            query_text=f"Targeted retrieval for missing zones: {', '.join(missing_zones)}",
            target_section_types=focused_sections,
            risk_profile=RiskProfile.HIGH_RISK,  # Use high risk for patches
            max_tokens_budget=min(2000, intent.max_tokens_budget // 2),
            required_structural_zones=missing_zones
        )
        
        # Retrieve candidates for missing zones
        candidates = await self.retriever.retrieve_candidates(focused_intent, top_k=20)
        
        # Rank and select patches
        ranked_set = await self.ranker.rank_candidates(candidates, focused_intent)
        
        # Create patch package
        packager = ContextPackager()
        patch_package = packager.create_package(ranked_set, focused_intent)
        
        # Merge patches into original package
        return self._merge_packages(context_package, patch_package)
    
    def _map_zones_to_sections(self, zones: List[str]) -> List[SectionType]:
        """Map zone names to section types"""
        zone_to_section = {
            "front_summary": [SectionType.TITLE_SUMMARY],
            "instruments_register": [SectionType.INSTRUMENTS_REGISTER],
            "caveats_section": [SectionType.CAVEATS_SECTION],
            "reserve_fund_report": [SectionType.RESERVE_FUND_REPORT],
            "financial_statements": [SectionType.FINANCIAL_STATEMENTS],
            "special_resolutions": [SectionType.SPECIAL_RESOLUTIONS],
            "tax_certificate": [SectionType.TAX_CERTIFICATE],
            "municipal_arrears": [SectionType.TAX_ARREARS],
            "tax_liens": [SectionType.TAX_ARREARS]
        }
        
        sections = []
        for zone in zones:
            sections.extend(zone_to_section.get(zone, []))
        
        return list(set(sections))  # Remove duplicates
    
    def _merge_packages(
        self,
        original: ContextPackage,
        patch: ContextPackage
    ) -> ContextPackage:
        """Merge patch package into original"""
        # Combine chunks, maintaining order
        all_chunks = list(original.ordered_chunks)
        
        # Add patch chunks that aren't already present
        existing_chunk_ids = {chunk.chunk_id for chunk in all_chunks}
        
        for patch_chunk in patch.ordered_chunks:
            if patch_chunk.chunk_id not in existing_chunk_ids:
                all_chunks.append(patch_chunk)
        
        # Update structural TOC
        combined_toc = f"{original.structural_toc} | PATCHED: {patch.structural_toc}"
        
        # Update exclusions note
        combined_exclusions = f"{original.exclusions_note}; {patch.exclusions_note}"
        
        return ContextPackage(
            deal_id=original.deal_id,
            intent_id=original.intent_id,
            agent_id=original.agent_id,
            ordered_chunks=all_chunks,
            structural_toc=combined_toc,
            exclusions_note=combined_exclusions,
            total_tokens=original.total_tokens + patch.total_tokens,
            risk_summary=f"{original.risk_summary} + {patch.risk_summary}"
        )

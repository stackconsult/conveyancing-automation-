"""
Stage 1 Retrieval System - LangGraph Integration

This module contains the LangGraph node interface and agent adapters
for integrating the RetrievalAgent into the conveyancing workflow.
"""

from typing import Dict, List, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

from ..schemas.core_schemas import (
    RetrievalIntent, ContextPackage, RetrievalSummary,
    SectionType, RiskProfile, create_retrieval_intent
)
from .retrieval_agent import RetrievalAgent

# ============================================================================
# LANGGRAPH NODE INTERFACE
# ============================================================================

@dataclass
class DealState:
    """
    State object that flows through LangGraph nodes.
    Contains all deal information and intermediate results.
    """
    deal_id: str
    current_agent: str
    current_step: str
    documents_processed: List[str]
    context_packages: Dict[str, ContextPackage]
    retrieval_summaries: Dict[str, RetrievalSummary]
    analysis_results: Dict[str, Any]
    error_log: List[str]
    
    def get_context(self, step: str) -> Optional[ContextPackage]:
        """Get context package for a specific step"""
        return self.context_packages.get(step)
    
    def add_context(self, step: str, context: ContextPackage):
        """Add context package for a step"""
        self.context_packages[step] = context
    
    def add_summary(self, step: str, summary: RetrievalSummary):
        """Add retrieval summary for a step"""
        self.retrieval_summaries[step] = summary
    
    def add_error(self, error: str):
        """Add error to error log"""
        self.error_log.append(f"[{datetime.utcnow().isoformat()}] {error}")

class RetrievalNode:
    """
    LangGraph node for Stage 1 retrieval operations.
    
    This node integrates the RetrievalAgent into the LangGraph
    workflow, handling state management and intent building.
    """
    
    def __init__(self, retrieval_agent: RetrievalAgent):
        """
        Initialize the retrieval node.
        
        Args:
            retrieval_agent: Configured RetrievalAgent instance
        """
        self.retrieval_agent = retrieval_agent
        self.adapters = {
            "investigator": InvestigatorAdapter(),
            "tax": TaxAdapter(),
            "scribe": ScribeAdapter()
        }
    
    async def __call__(self, state: DealState) -> DealState:
        """
        Execute retrieval for the current agent and step.
        
        Args:
            state: Current deal state
            
        Returns:
            Updated deal state with context and summary
        """
        try:
            # Build intent from current state and agent
            intent = self._build_intent_from_state(state)
            
            # Execute retrieval
            context_package, summary = await self.retrieval_agent.retrieve(intent)
            
            # Update state
            state.add_context(state.current_step, context_package)
            state.add_summary(state.current_step, summary)
            
            # Log successful retrieval
            print(f"Retrieval completed for {state.current_agent}/{state.current_step}")
            print(f"  Chunks: {summary.chunks_selected}, Tokens: {summary.tokens_selected}")
            print(f"  Status: {summary.status.value}, Confidence: {summary.confidence_score:.2f}")
            
            return state
            
        except Exception as e:
            error_msg = f"Retrieval failed for {state.current_agent}/{state.current_step}: {str(e)}"
            state.add_error(error_msg)
            print(f"ERROR: {error_msg}")
            return state
    
    def _build_intent_from_state(self, state: DealState) -> RetrievalIntent:
        """
        Build retrieval intent from current state.
        
        Args:
            state: Current deal state
            
        Returns:
            RetrievalIntent for the current agent
        """
        adapter = self.adapters.get(state.current_agent.lower())
        
        if not adapter:
            raise ValueError(f"No adapter found for agent: {state.current_agent}")
        
        return adapter.build_intent(state)

# ============================================================================
# AGENT ADAPTERS
# ============================================================================

class AgentAdapter:
    """Base class for agent-specific intent builders"""
    
    def build_intent(self, state: DealState) -> RetrievalIntent:
        """Build retrieval intent from deal state"""
        raise NotImplementedError("Subclasses must implement build_intent")

class InvestigatorAdapter(AgentAdapter):
    """Adapter for Investigator agent - focuses on title risks"""
    
    def build_intent(self, state: DealState) -> RetrievalIntent:
        """Build intent for title risk scanning"""
        return create_retrieval_intent(
            deal_id=state.deal_id,
            agent_id="investigator_r1",
            query="Identify title risks, encumbrances, ownership issues, and potential problems",
            sections=[
                SectionType.TITLE_SUMMARY,
                SectionType.INSTRUMENTS_REGISTER,
                SectionType.CAVEATS_SECTION,
                SectionType.LEGAL_DESCRIPTION
            ],
            risk_profile=RiskProfile.HIGH_RISK,
            max_tokens=4000,
            required_structural_zones=[
                "front_summary",
                "instruments_register", 
                "caveats_section"
            ]
        )

class TaxAdapter(AgentAdapter):
    """Adapter for Tax agent - focuses on tax compliance"""
    
    def build_intent(self, state: DealState) -> RetrievalIntent:
        """Build intent for tax arrears and compliance checking"""
        return create_retrieval_intent(
            deal_id=state.deal_id,
            agent_id="tax_r1",
            query="Check for tax arrears, penalties, municipal charges, and tax compliance issues",
            sections=[
                SectionType.TAX_CERTIFICATE,
                SectionType.TAX_ARREARS,
                SectionType.FINANCIAL_STATEMENTS
            ],
            risk_profile=RiskProfile.BALANCED,
            max_tokens=2000,
            required_structural_zones=[
                "tax_certificate",
                "municipal_arrears"
            ]
        )

class ScribeAdapter(AgentAdapter):
    """Adapter for Scribe agent - focuses on document drafting"""
    
    def build_intent(self, state: DealState) -> RetrievalIntent:
        """Build intent for precedent alignment and drafting support"""
        return create_retrieval_intent(
            deal_id=state.deal_id,
            agent_id="scribe_r1",
            query="Find precedents, standard clauses, and drafting patterns for document preparation",
            sections=[
                SectionType.PURCHASE_AGREEMENT,
                SectionType.GENERAL,
                SectionType.LEGAL_DESCRIPTION
            ],
            risk_profile=RiskProfile.LOW_RISK,
            max_tokens=3000,
            required_structural_zones=[
                "purchase_agreement_terms"
            ]
        )

class CondoAdapter(AgentAdapter):
    """Adapter for Condo specialist - focuses on condominium governance"""
    
    def build_intent(self, state: DealState) -> RetrievalIntent:
        """Build intent for condo governance and financial health"""
        return create_retrieval_intent(
            deal_id=state.deal_id,
            agent_id="condo_r1",
            query="Analyze condo governance, financial health, reserve fund status, and special assessments",
            sections=[
                SectionType.CONDO_BYLAWS,
                SectionType.CONDO_MINUTES,
                SectionType.RESERVE_FUND_REPORT,
                SectionType.FINANCIAL_STATEMENTS,
                SectionType.SPECIAL_RESOLUTIONS
            ],
            risk_profile=RiskProfile.HIGH_RISK,
            max_tokens=5000,
            required_structural_zones=[
                "reserve_fund_report",
                "financial_statements",
                "special_resolutions"
            ]
        )

# ============================================================================
# WORKFLOW INTEGRATION
# ============================================================================

class RetrievalWorkflow:
    """
    High-level workflow orchestrator for retrieval operations.
    Manages multiple retrieval steps across different agents.
    """
    
    def __init__(self, retrieval_agent: RetrievalAgent):
        self.retrieval_agent = retrieval_agent
        self.node = RetrievalNode(retrieval_agent)
    
    async def process_deal(
        self,
        deal_id: str,
        agents: List[str],
        documents: List[str]
    ) -> DealState:
        """
        Process a complete deal through multiple retrieval steps.
        
        Args:
            deal_id: Deal identifier
            agents: List of agents to run
            documents: List of processed documents
            
        Returns:
            Final deal state with all contexts and summaries
        """
        # Initialize state
        state = DealState(
            deal_id=deal_id,
            current_agent="",
            current_step="",
            documents_processed=documents,
            context_packages={},
            retrieval_summaries={},
            analysis_results={},
            error_log=[]
        )
        
        # Process each agent
        for agent in agents:
            state.current_agent = agent
            state.current_step = f"{agent}_retrieval"
            
            # Execute retrieval
            state = await self.node(state)
            
            # Check for critical errors
            if state.error_log and any("FAILED" in error for error in state.error_log):
                print(f"Critical error in {agent}, stopping workflow")
                break
        
        return state
    
    async def run_single_agent(
        self,
        deal_id: str,
        agent: str,
        query: str,
        sections: List[str],
        risk_profile: str = "BALANCED",
        max_tokens: int = 4000
    ) -> Tuple[ContextPackage, RetrievalSummary]:
        """
        Run retrieval for a single agent with custom parameters.
        
        Args:
            deal_id: Deal identifier
            agent: Agent name
            query: Custom query
            sections: Target sections
            risk_profile: Risk tolerance
            max_tokens: Token budget
            
        Returns:
            Tuple of (context_package, retrieval_summary)
        """
        from ..schemas.core_schemas import SectionType, RiskProfile
        
        # Convert inputs
        section_types = [SectionType(st) for st in sections]
        risk_enum = RiskProfile(risk_profile)
        
        # Create custom intent
        intent = create_retrieval_intent(
            deal_id=deal_id,
            agent_id=f"{agent}_r1",
            query=query,
            sections=section_types,
            risk_profile=risk_enum,
            max_tokens=max_tokens
        )
        
        # Execute retrieval
        return await self.retrieval_agent.retrieve(intent)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_retrieval_workflow(
    mem0_client: Any,
    vector_client: Any,
    embedding_model: Any
) -> RetrievalWorkflow:
    """
    Create a complete retrieval workflow.
    
    Args:
        mem0_client: Mem0 client instance
        vector_client: Vector database client instance
        embedding_model: Embedding model instance
        
    Returns:
        Configured RetrievalWorkflow
    """
    from .retrieval_agent import create_retrieval_agent
    
    # Create retrieval agent
    retrieval_agent = create_retrieval_agent(
        mem0_client=mem0_client,
        vector_client=vector_client,
        embedding_model=embedding_model
    )
    
    # Create workflow
    return RetrievalWorkflow(retrieval_agent)

async def run_standard_conveyancing_retrieval(
    workflow: RetrievalWorkflow,
    deal_id: str,
    documents: List[str]
) -> DealState:
    """
    Run the standard conveyancing retrieval workflow.
    
    Args:
        workflow: Configured retrieval workflow
        deal_id: Deal identifier
        documents: List of processed documents
        
    Returns:
        Deal state with all retrieval results
    """
    # Standard agent order for conveyancing
    agents = ["investigator", "tax", "scribe"]
    
    # If condo documents are present, add condo specialist
    if any("condo" in doc.lower() for doc in documents):
        agents.insert(2, "condo")  # Insert after tax, before scribe
    
    return await workflow.process_deal(deal_id, agents, documents)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """
    Example of how to use the retrieval system.
    """
    # This would be replaced with actual client initialization
    mem0_client = None  # Your Mem0 client
    vector_client = None  # Your vector database client
    embedding_model = None  # Your embedding model
    
    # Create workflow
    workflow = create_retrieval_workflow(
        mem0_client=mem0_client,
        vector_client=vector_client,
        embedding_model=embedding_model
    )
    
    # Process a deal
    deal_id = "deal_12345"
    documents = ["purchase_agreement.pdf", "title_search.pdf", "condo_docs.pdf"]
    
    state = await run_standard_conveyancing_retrieval(
        workflow=workflow,
        deal_id=deal_id,
        documents=documents
    )
    
    # Access results
    for step, context in state.context_packages.items():
        print(f"Step {step}: {context.chunk_count} chunks, {context.total_tokens} tokens")
    
    for step, summary in state.retrieval_summaries.items():
        print(f"Step {step}: {summary.status.value}, confidence {summary.confidence_score:.2f}")

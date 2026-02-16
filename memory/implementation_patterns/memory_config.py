# Conveyancing Memory Integration Configuration

"""
Configuration and setup for memory integration in the conveyancing automation system.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from mem0 import MemoryClient

@dataclass
class MemoryConfig:
    """Configuration for memory integration."""
    
    api_key: str
    project_id: str
    base_url: str = "https://api.mem0.ai"
    
    # Memory categories for conveyancing
    legal_knowledge_categories: List[str] = None
    document_template_categories: List[str] = None
    process_workflow_categories: List[str] = None
    stakeholder_categories: List[str] = None
    risk_factor_categories: List[str] = None
    compliance_categories: List[str] = None
    market_data_categories: List[str] = None
    
    def __post_init__(self):
        """Initialize default categories if not provided."""
        if self.legal_knowledge_categories is None:
            self.legal_knowledge_categories = [
                "property_laws", "contract_law", "compliance_frameworks", 
                "legal_precedents", "jurisdiction_requirements"
            ]
        
        if self.document_template_categories is None:
            self.document_template_categories = [
                "purchase_agreements", "deed_templates", "disclosure_forms",
                "legal_documents", "standard_forms"
            ]
        
        if self.process_workflow_categories is None:
            self.process_workflow_categories = [
                "conveyancing_procedures", "compliance_workflows", 
                "closing_processes", "document_processing", "coordination_flows"
            ]
        
        if self.stakeholder_categories is None:
            self.stakeholder_categories = [
                "client_preferences", "agent_coordination", "authority_interactions",
                "third_party_services", "communication_protocols"
            ]
        
        if self.risk_factor_categories is None:
            self.risk_factor_categories = [
                "legal_risks", "financial_risks", "timeline_risks",
                "market_risks", "compliance_risks"
            ]
        
        if self.compliance_categories is None:
            self.compliance_categories = [
                "regulatory_requirements", "validation_rules", "audit_requirements",
                "data_protection", "industry_standards"
            ]
        
        if self.market_data_categories is None:
            self.market_data_categories = [
                "property_valuations", "market_trends", "regional_statistics",
                "transaction_history", "economic_indicators"
            ]

class ConveyancingMemoryManager:
    """Main memory manager for conveyancing automation system."""
    
    def __init__(self, config: MemoryConfig):
        """
        Initialize the memory manager.
        
        Args:
            config: Memory configuration
        """
        self.config = config
        self.client = MemoryClient(api_key=config.api_key)
        self.project_id = config.project_id
        
    async def initialize_memory_categories(self) -> Dict[str, bool]:
        """
        Initialize all memory categories with base content.
        
        Returns:
            Dictionary indicating success/failure for each category
        """
        initialization_results = {}
        
        # Initialize legal knowledge base
        initialization_results["legal_knowledge"] = await self._initialize_legal_knowledge()
        
        # Initialize document templates
        initialization_results["document_templates"] = await self._initialize_document_templates()
        
        # Initialize process workflows
        initialization_results["process_workflows"] = await self._initialize_process_workflows()
        
        # Initialize stakeholder data patterns
        initialization_results["stakeholder_data"] = await self._initialize_stakeholder_data()
        
        # Initialize risk factors
        initialization_results["risk_factors"] = await self._initialize_risk_factors()
        
        # Initialize compliance rules
        initialization_results["compliance_rules"] = await self._initialize_compliance_rules()
        
        # Initialize market data structure
        initialization_results["market_data"] = await self._initialize_market_data()
        
        return initialization_results
    
    async def _initialize_legal_knowledge(self) -> bool:
        """Initialize legal knowledge base."""
        try:
            # Add property law fundamentals
            await self.client.add(
                user_id="system",
                messages=[
                    {
                        "role": "system",
                        "content": "Property law fundamentals: Real property includes land and anything permanently attached to it. Property rights include ownership, possession, use, enjoyment, exclusion, and disposition. Title transfers require deed recording and proper execution."
                    }
                ],
                metadata={
                    "category": "legal_knowledge",
                    "subcategory": "property_laws",
                    "jurisdiction": "general",
                    "memory_type": "fundamental_principle",
                    "priority": "high"
                }
            )
            
            # Add contract law essentials
            await self.client.add(
                user_id="system",
                messages=[
                    {
                        "role": "system",
                        "content": "Contract law essentials: Valid contracts require offer, acceptance, consideration, capacity, and legal purpose. Real estate contracts must be in writing (Statute of Frauds) and include essential terms like property description, price, and parties."
                    }
                ],
                metadata={
                    "category": "legal_knowledge",
                    "subcategory": "contract_law",
                    "jurisdiction": "general",
                    "memory_type": "fundamental_principle",
                    "priority": "high"
                }
            )
            
            return True
        except Exception as e:
            print(f"Error initializing legal knowledge: {e}")
            return False
    
    async def _initialize_document_templates(self) -> bool:
        """Initialize document templates."""
        try:
            # Add purchase agreement template structure
            await self.client.add(
                user_id="system",
                messages=[
                    {
                        "role": "system",
                        "content": "Purchase agreement template must include: parties identification, property description, purchase price, earnest money, financing contingency, inspection contingency, title contingency, closing date, possession date, and signatures."
                    }
                ],
                metadata={
                    "category": "document_templates",
                    "subcategory": "purchase_agreements",
                    "template_type": "required_elements",
                    "priority": "high"
                }
            )
            
            # Add deed template requirements
            await self.client.add(
                user_id="system",
                messages=[
                    {
                        "role": "system",
                        "content": "Deed template requirements: Grantor and grantee names, legal property description, consideration statement, granting words (grant, convey, transfer), signature of grantor, notarization, and recording information."
                    }
                ],
                metadata={
                    "category": "document_templates",
                    "subcategory": "deed_templates",
                    "template_type": "required_elements",
                    "priority": "high"
                }
            )
            
            return True
        except Exception as e:
            print(f"Error initializing document templates: {e}")
            return False
    
    async def _initialize_process_workflows(self) -> bool:
        """Initialize process workflows."""
        try:
            # Add standard conveyancing workflow
            await self.client.add(
                user_id="system",
                messages=[
                    {
                        "role": "system",
                        "content": "Standard conveyancing workflow: 1) Pre-contract (title search, property disclosure), 2) Contract negotiation and execution, 3) Due diligence period (inspections, financing), 4) Pre-closing (title commitment, final walkthrough), 5) Closing (document signing, fund transfer), 6) Post-closing (recording, possession)."
                    }
                ],
                metadata={
                    "category": "process_workflows",
                    "subcategory": "conveyancing_procedures",
                    "workflow_type": "standard_process",
                    "priority": "high"
                }
            )
            
            return True
        except Exception as e:
            print(f"Error initializing process workflows: {e}")
            return False
    
    async def _initialize_stakeholder_data(self) -> bool:
        """Initialize stakeholder data patterns."""
        try:
            # Add client preference patterns
            await self.client.add(
                user_id="system",
                messages=[
                    {
                        "role": "system",
                        "content": "Client preference tracking: Communication methods (email, phone, portal), meeting preferences (in-person, virtual), document delivery preferences, timeline priorities, and special requirements or accommodations."
                    }
                ],
                metadata={
                    "category": "stakeholder_data",
                    "subcategory": "client_preferences",
                    "data_type": "preference_patterns",
                    "priority": "medium"
                }
            )
            
            return True
        except Exception as e:
            print(f"Error initializing stakeholder data: {e}")
            return False
    
    async def _initialize_risk_factors(self) -> bool:
        """Initialize risk factors."""
        try:
            # Add common legal risks
            await self.client.add(
                user_id="system",
                messages=[
                    {
                        "role": "system",
                        "content": "Common legal risks in conveyancing: Title defects (liens, encumbrances), boundary disputes, zoning violations, undisclosed easements, contract contingencies, financing failures, inspection issues, and closing delays."
                    }
                ],
                metadata={
                    "category": "risk_factors",
                    "subcategory": "legal_risks",
                    "risk_type": "common_issues",
                    "priority": "high"
                }
            )
            
            return True
        except Exception as e:
            print(f"Error initializing risk factors: {e}")
            return False
    
    async def _initialize_compliance_rules(self) -> bool:
        """Initialize compliance rules."""
        try:
            # Add general compliance requirements
            await self.client.add(
                user_id="system",
                messages=[
                    {
                        "role": "system",
                        "content": "General compliance requirements: Proper licensing, disclosure obligations, trust account handling, document retention, privacy protection, anti-money laundering checks, and ethical conduct standards."
                    }
                ],
                metadata={
                    "category": "compliance_rules",
                    "subcategory": "regulatory_requirements",
                    "compliance_type": "general_standards",
                    "priority": "high"
                }
            )
            
            return True
        except Exception as e:
            print(f"Error initializing compliance rules: {e}")
            return False
    
    async def _initialize_market_data(self) -> bool:
        """Initialize market data structure."""
        try:
            # Add market data categories
            await self.client.add(
                user_id="system",
                messages=[
                    {
                        "role": "system",
                        "content": "Market data categories: Property values (median, average, price per sq ft), market trends (appreciation rates, inventory levels), regional statistics (days on market, sale-to-list ratio), economic indicators (interest rates, employment rates)."
                    }
                ],
                metadata={
                    "category": "market_data",
                    "subcategory": "market_trends",
                    "data_type": "category_structure",
                    "priority": "medium"
                }
            )
            
            return True
        except Exception as e:
            print(f"Error initializing market data: {e}")
            return False
    
    async def search_case_memories(self, case_id: str, query: str = "") -> List[Dict]:
        """
        Search all memories related to a specific case.
        
        Args:
            case_id: Unique case identifier
            query: Optional search query to filter results
            
        Returns:
            List of relevant memories
        """
        try:
            search_query = f"case {case_id} {query}".strip()
            
            memories = await self.client.search(
                query=search_query,
                filters={"case_id": case_id},
                top_k=20
            )
            
            return memories
        except Exception as e:
            print(f"Error searching case memories: {e}")
            return []
    
    async def add_case_memory(self, case_id: str, content: str, metadata: Dict) -> Dict:
        """
        Add a memory related to a specific case.
        
        Args:
            case_id: Unique case identifier
            content: Memory content
            metadata: Additional metadata
            
        Returns:
            Created memory response
        """
        try:
            memory_data = {
                "user_id": f"case_{case_id}",
                "messages": [{"role": "system", "content": content}],
                "metadata": {
                    "case_id": case_id,
                    "timestamp": datetime.now().isoformat(),
                    **metadata
                }
            }
            
            result = await self.client.add(**memory_data)
            return result
        except Exception as e:
            print(f"Error adding case memory: {e}")
            return {"error": str(e)}
    
    async def get_jurisdiction_requirements(self, jurisdiction: str) -> List[Dict]:
        """
        Get specific requirements for a jurisdiction.
        
        Args:
            jurisdiction: Jurisdiction name (e.g., "texas", "california")
            
        Returns:
            List of jurisdiction-specific requirements
        """
        try:
            requirements = await self.client.search(
                query=f"jurisdiction requirements {jurisdiction}",
                filters={
                    "category": "compliance_rules",
                    "jurisdiction": jurisdiction.lower()
                },
                top_k=10
            )
            
            return requirements
        except Exception as e:
            print(f"Error getting jurisdiction requirements: {e}")
            return []

# Factory function for easy initialization
def create_conveyancing_memory_manager(api_key: str, project_id: str) -> ConveyancingMemoryManager:
    """
    Create a configured conveyancing memory manager.
    
    Args:
        api_key: Mem0 API key
        project_id: Project identifier
        
    Returns:
        Configured memory manager
    """
    config = MemoryConfig(api_key=api_key, project_id=project_id)
    return ConveyancingMemoryManager(config)

# Environment-based initialization
def initialize_from_environment() -> Optional[ConveyancingMemoryManager]:
    """
    Initialize memory manager from environment variables.
    
    Expected environment variables:
    - MEM0_API_KEY: Mem0 API key
    - CONVEYANCING_PROJECT_ID: Project identifier
    
    Returns:
        Configured memory manager or None if environment not set
    """
    api_key = os.getenv("MEM0_API_KEY")
    project_id = os.getenv("CONVEYANCING_PROJECT_ID")
    
    if not api_key or not project_id:
        print("Missing required environment variables: MEM0_API_KEY, CONVEYANCING_PROJECT_ID")
        return None
    
    return create_conveyancing_memory_manager(api_key, project_id)

# Usage example
async def example_usage():
    """Example of how to use the conveyancing memory manager."""
    
    # Initialize from environment
    memory_manager = initialize_from_environment()
    if not memory_manager:
        return
    
    # Initialize memory categories
    init_results = await memory_manager.initialize_memory_categories()
    print(f"Initialization results: {init_results}")
    
    # Add case-specific memory
    case_memory = await memory_manager.add_case_memory(
        case_id="case_12345",
        content="Client requested closing date extension from Dec 15 to Dec 22 due to travel schedule.",
        metadata={
            "interaction_type": "client_request",
            "priority": "high",
            "category": "stakeholder_data"
        }
    )
    print(f"Added case memory: {case_memory}")
    
    # Search case memories
    case_memories = await memory_manager.search_case_memories("case_12345", "client request")
    print(f"Found {len(case_memories)} case memories")
    
    # Get jurisdiction requirements
    tx_requirements = await memory_manager.get_jurisdiction_requirements("texas")
    print(f"Found {len(tx_requirements)} Texas requirements")

if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    
    asyncio.run(example_usage())

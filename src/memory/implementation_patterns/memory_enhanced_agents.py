# Memory-Enhanced Agent Base Class

"""
Base class for memory-enhanced agents in the conveyancing automation system.
Provides common memory integration functionality for all agent types.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from mem0 import MemoryClient

class MemoryEnhancedAgent:
    """Base class for memory-enhanced conveyancing agents."""
    
    def __init__(self, agent_id: str, memory_client: MemoryClient, memory_categories: List[str]):
        """
        Initialize the memory-enhanced agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_client: Mem0 MemoryClient instance
            memory_categories: List of memory categories this agent uses
        """
        self.agent_id = agent_id
        self.memory_client = memory_client
        self.memory_categories = memory_categories
        self.case_context = {}
        
    async def load_contextual_memories(self, context: Dict[str, Any]) -> List[Dict]:
        """
        Load relevant memories based on current context.
        
        Args:
            context: Current case context including case_id, jurisdiction, etc.
            
        Returns:
            List of relevant memories
        """
        memories = []
        
        # Build search query from context
        search_query = self._build_search_query(context)
        
        # Search memories in each category
        for category in self.memory_categories:
            try:
                category_memories = await self.memory_client.search(
                    query=search_query,
                    filters={
                        "category": category,
                        "jurisdiction": context.get("jurisdiction", ""),
                        "case_id": context.get("case_id", "")
                    },
                    top_k=5
                )
                memories.extend(category_memories)
            except Exception as e:
                print(f"Error searching {category} memories: {e}")
                
        return memories
    
    def _build_search_query(self, context: Dict[str, Any]) -> str:
        """Build search query from context."""
        query_parts = []
        
        if context.get("case_id"):
            query_parts.append(f"case {context['case_id']}")
        if context.get("property_address"):
            query_parts.append(f"property {context['property_address']}")
        if context.get("document_type"):
            query_parts.append(f"document {context['document_type']}")
        if context.get("jurisdiction"):
            query_parts.append(f"jurisdiction {context['jurisdiction']}")
            
        return " ".join(query_parts)
    
    async def store_interaction_memory(self, interaction_data: Dict[str, Any]) -> Dict:
        """
        Store important interactions for future reference.
        
        Args:
            interaction_data: Dictionary containing interaction details
            
        Returns:
            Created memory response
        """
        memory_content = {
            "messages": interaction_data.get("messages", []),
            "user_id": interaction_data.get("user_id", "system"),
            "metadata": {
                "agent_id": self.agent_id,
                "interaction_type": interaction_data.get("type", "general"),
                "case_id": interaction_data.get("case_id"),
                "jurisdiction": interaction_data.get("jurisdiction"),
                "timestamp": datetime.now().isoformat(),
                "priority": interaction_data.get("priority", "normal"),
                "document_type": interaction_data.get("document_type"),
                "stakeholder": interaction_data.get("stakeholder")
            }
        }
        
        try:
            result = await self.memory_client.add(**memory_content)
            return result
        except Exception as e:
            print(f"Error storing interaction memory: {e}")
            return {"error": str(e)}
    
    async def update_case_context(self, context: Dict[str, Any]) -> None:
        """Update the agent's case context."""
        self.case_context.update(context)
    
    async def get_relevant_precedents(self, case_context: Dict[str, Any]) -> List[Dict]:
        """Get relevant legal precedents for the current case."""
        try:
            precedents = await self.memory_client.search(
                query=f"legal precedent {case_context.get('jurisdiction', '')} {case_context.get('case_type', '')}",
                filters={
                    "category": "legal_knowledge",
                    "memory_type": "precedent",
                    "jurisdiction": case_context.get("jurisdiction", "")
                },
                top_k=3
            )
            return precedents
        except Exception as e:
            print(f"Error retrieving precedents: {e}")
            return []
    
    async def get_compliance_requirements(self, case_context: Dict[str, Any]) -> List[Dict]:
        """Get compliance requirements for the current case."""
        try:
            requirements = await self.memory_client.search(
                query=f"compliance requirements {case_context.get('jurisdiction', '')} {case_context.get('transaction_type', '')}",
                filters={
                    "category": "compliance_rules",
                    "jurisdiction": case_context.get("jurisdiction", ""),
                    "transaction_type": case_context.get("transaction_type", "")
                },
                top_k=10
            )
            return requirements
        except Exception as e:
            print(f"Error retrieving compliance requirements: {e}")
            return []

class DocumentAnalysisAgent(MemoryEnhancedAgent):
    """Agent specialized in legal document analysis with memory enhancement."""
    
    def __init__(self, memory_client: MemoryClient):
        super().__init__(
            agent_id="document_analysis_agent",
            memory_client=memory_client,
            memory_categories=["legal_knowledge", "document_templates", "compliance_rules"]
        )
    
    async def analyze_document(self, document_content: str, case_context: Dict[str, Any]) -> Dict:
        """
        Analyze legal document with memory-enhanced context.
        
        Args:
            document_content: The document text to analyze
            case_context: Current case context
            
        Returns:
            Analysis results with memory references
        """
        # Load relevant memories
        relevant_memories = await self.load_contextual_memories(case_context)
        
        # Get relevant precedents
        precedents = await self.get_relevant_precedents(case_context)
        
        # Get compliance requirements
        compliance_requirements = await self.get_compliance_requirements(case_context)
        
        # Perform analysis (placeholder for actual analysis logic)
        analysis_result = {
            "document_type": self._identify_document_type(document_content),
            "key_terms": self._extract_key_terms(document_content),
            "compliance_issues": self._check_compliance(document_content, compliance_requirements),
            "relevant_precedents": precedents,
            "memory_references": relevant_memories,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Store analysis memory
        await self.store_interaction_memory({
            "messages": [
                {
                    "role": "user",
                    "content": f"Document analysis completed for {case_context.get('case_id')}"
                },
                {
                    "role": "assistant",
                    "content": f"Analysis identified {analysis_result['document_type']} with {len(analysis_result['compliance_issues'])} compliance issues"
                }
            ],
            "type": "document_analysis",
            "case_id": case_context.get("case_id"),
            "jurisdiction": case_context.get("jurisdiction"),
            "document_type": analysis_result["document_type"]
        })
        
        return analysis_result
    
    def _identify_document_type(self, content: str) -> str:
        """Identify document type from content."""
        content_lower = content.lower()
        
        if "purchase agreement" in content_lower or "sales contract" in content_lower:
            return "purchase_agreement"
        elif "deed" in content_lower and "transfer" in content_lower:
            return "transfer_deed"
        elif "disclosure" in content_lower:
            return "disclosure_statement"
        elif "title" in content_lower and "search" in content_lower:
            return "title_report"
        else:
            return "unknown"
    
    def _extract_key_terms(self, content: str) -> List[str]:
        """Extract key legal terms from document."""
        # Placeholder implementation - would use NLP in production
        key_terms = []
        legal_terms = ["purchase price", "closing date", "property address", "buyer", "seller", "earnest money", "contingencies"]
        
        for term in legal_terms:
            if term.lower() in content.lower():
                key_terms.append(term)
        
        return key_terms
    
    def _check_compliance(self, content: str, requirements: List[Dict]) -> List[Dict]:
        """Check document compliance against requirements."""
        compliance_issues = []
        
        for requirement in requirements:
            # Placeholder compliance checking logic
            if "disclosure" in requirement.get("memory", "").lower() and "disclosure" not in content.lower():
                compliance_issues.append({
                    "requirement": requirement["memory"],
                    "status": "missing",
                    "severity": "high"
                })
        
        return compliance_issues

class ComplianceAgent(MemoryEnhancedAgent):
    """Agent specialized in regulatory compliance with memory enhancement."""
    
    def __init__(self, memory_client: MemoryClient):
        super().__init__(
            agent_id="compliance_agent",
            memory_client=memory_client,
            memory_categories=["compliance_rules", "risk_factors", "legal_knowledge"]
        )
    
    async def validate_compliance(self, case_data: Dict[str, Any]) -> Dict:
        """
        Validate case compliance using memory-enhanced rules.
        
        Args:
            case_data: Case data to validate
            
        Returns:
            Compliance validation results
        """
        # Load relevant compliance rules
        relevant_memories = await self.load_contextual_memories(case_data)
        
        # Get risk factors
        risk_factors = await self.memory_client.search(
            query=f"risk factors {case_data.get('jurisdiction', '')} {case_data.get('transaction_type', '')}",
            filters={
                "category": "risk_factors",
                "jurisdiction": case_data.get("jurisdiction", "")
            },
            top_k=5
        )
        
        # Perform compliance validation
        validation_result = {
            "compliance_status": "pending",
            "checked_requirements": [],
            "identified_risks": risk_factors,
            "memory_references": relevant_memories,
            "validation_timestamp": datetime.now().isoformat()
        }
        
        # Store validation memory
        await self.store_interaction_memory({
            "messages": [
                {
                    "role": "user",
                    "content": f"Compliance validation initiated for case {case_data.get('case_id')}"
                },
                {
                    "role": "assistant",
                    "content": f"Validation checking {len(relevant_memories)} requirements and {len(risk_factors)} risk factors"
                }
            ],
            "type": "compliance_validation",
            "case_id": case_data.get("case_id"),
            "jurisdiction": case_data.get("jurisdiction")
        })
        
        return validation_result

# Usage example
async def main():
    """Example usage of memory-enhanced agents."""
    
    # Initialize memory client
    memory_client = MemoryClient(api_key="your_mem0_api_key")
    
    # Create agents
    doc_agent = DocumentAnalysisAgent(memory_client)
    compliance_agent = ComplianceAgent(memory_client)
    
    # Example case context
    case_context = {
        "case_id": "case_12345",
        "jurisdiction": "texas",
        "property_address": "123 Main Street, Austin, TX",
        "transaction_type": "residential_sale"
    }
    
    # Analyze document
    document_content = """
    PURCHASE AGREEMENT
    
    This Purchase Agreement is made on November 1, 2023 between John Smith (Buyer) and Jane Doe (Seller) for the property located at 123 Main Street, Austin, TX.
    
    Purchase Price: $450,000
    Closing Date: December 15, 2023
    Earnest Money: $5,000
    
    The property is sold in "as-is" condition.
    """
    
    analysis_result = await doc_agent.analyze_document(document_content, case_context)
    print(f"Document Analysis: {analysis_result['document_type']}")
    
    # Validate compliance
    compliance_result = await compliance_agent.validate_compliance(case_context)
    print(f"Compliance Status: {compliance_result['compliance_status']}")

if __name__ == "__main__":
    asyncio.run(main())

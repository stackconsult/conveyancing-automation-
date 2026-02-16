# Memory-Enhanced Orchestrator

"""
Enhanced orchestrator for the conveyancing automation system with memory integration.
Provides intelligent coordination between agents using persistent memory.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from mem0 import MemoryClient

from memory.enhanced_agents import (
    MemoryEnhancedAgent, 
    DocumentAnalysisAgent, 
    ComplianceAgent
)
from memory.memory_config import ConveyancingMemoryManager

@dataclass
class CaseContext:
    """Context information for a conveyancing case."""
    case_id: str
    client_id: str
    property_address: str
    jurisdiction: str
    transaction_type: str
    status: str = "active"
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

@dataclass
class AgentTask:
    """Task definition for agent execution."""
    task_id: str
    agent_type: str
    case_context: CaseContext
    input_data: Dict[str, Any]
    priority: str = "normal"
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class MemoryAwareOrchestrator:
    """Memory-enhanced orchestrator for conveyancing automation."""
    
    def __init__(self, memory_manager: ConveyancingMemoryManager):
        """
        Initialize the memory-aware orchestrator.
        
        Args:
            memory_manager: Configured memory manager
        """
        self.memory_manager = memory_manager
        self.agents = {}
        self.active_cases = {}
        self.task_queue = []
        self.completed_tasks = {}
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all available agents."""
        self.agents = {
            "document_analysis": DocumentAnalysisAgent(self.memory_manager.client),
            "compliance": ComplianceAgent(self.memory_manager.client),
            # Additional agents can be added here
        }
    
    async def process_conveyancing_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete conveyancing case with memory enhancement.
        
        Args:
            case_data: Dictionary containing case information
            
        Returns:
            Processing results with memory references
        """
        # Create case context
        case_context = CaseContext(
            case_id=case_data["case_id"],
            client_id=case_data["client_id"],
            property_address=case_data["property_address"],
            jurisdiction=case_data["jurisdiction"],
            transaction_type=case_data["transaction_type"]
        )
        
        # Store case context in memory
        await self._store_case_context(case_context)
        
        # Create agent pipeline tasks
        tasks = self._create_agent_pipeline(case_context, case_data)
        
        # Execute tasks with memory enhancement
        results = await self._execute_agent_pipeline(tasks)
        
        # Store processing summary in memory
        await self._store_processing_summary(case_context, results)
        
        return {
            "case_id": case_context.case_id,
            "status": "completed",
            "results": results,
            "processing_timestamp": datetime.now().isoformat()
        }
    
    async def _store_case_context(self, case_context: CaseContext):
        """Store case context in memory."""
        context_content = f"""
        Case Context:
        - Case ID: {case_context.case_id}
        - Client ID: {case_context.client_id}
        - Property: {case_context.property_address}
        - Jurisdiction: {case_context.jurisdiction}
        - Transaction Type: {case_context.transaction_type}
        - Status: {case_context.status}
        - Created: {case_context.created_at}
        """
        
        await self.memory_manager.add_case_memory(
            case_id=case_context.case_id,
            content=context_content,
            metadata={
                "memory_type": "case_context",
                "category": "process_workflows",
                "subcategory": "case_management"
            }
        )
    
    def _create_agent_pipeline(self, case_context: CaseContext, case_data: Dict[str, Any]) -> List[AgentTask]:
        """Create pipeline of agent tasks for the case."""
        tasks = []
        
        # Document Analysis Task
        if "documents" in case_data:
            tasks.append(AgentTask(
                task_id=f"doc_analysis_{case_context.case_id}",
                agent_type="document_analysis",
                case_context=case_context,
                input_data={
                    "documents": case_data["documents"],
                    "analysis_type": "comprehensive"
                },
                priority="high"
            ))
        
        # Compliance Validation Task
        tasks.append(AgentTask(
            task_id=f"compliance_{case_context.case_id}",
            agent_type="compliance",
            case_context=case_context,
            input_data={
                "transaction_data": case_data,
                "validation_level": "full"
            },
            priority="high",
            dependencies=[f"doc_analysis_{case_context.case_id}"] if "documents" in case_data else []
        ))
        
        return tasks
    
    async def _execute_agent_pipeline(self, tasks: List[AgentTask]) -> Dict[str, Any]:
        """Execute the agent pipeline with memory enhancement."""
        results = {}
        
        # Sort tasks by priority and dependencies
        sorted_tasks = self._sort_tasks_by_dependencies(tasks)
        
        for task in sorted_tasks:
            try:
                # Get agent instance
                agent = self.agents.get(task.agent_type)
                if not agent:
                    results[task.task_id] = {
                        "status": "error",
                        "error": f"Agent {task.agent_type} not found"
                    }
                    continue
                
                # Update agent context
                await agent.update_case_context(asdict(task.case_context))
                
                # Execute task based on agent type
                task_result = await self._execute_agent_task(agent, task)
                results[task.task_id] = task_result
                
                # Store task result in memory
                await self._store_task_result(task, task_result)
                
            except Exception as e:
                results[task.task_id] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    async def _execute_agent_task(self, agent: MemoryEnhancedAgent, task: AgentTask) -> Dict[str, Any]:
        """Execute a specific agent task."""
        if task.agent_type == "document_analysis":
            return await self._execute_document_analysis(agent, task)
        elif task.agent_type == "compliance":
            return await self._execute_compliance_validation(agent, task)
        else:
            return {
                "status": "error",
                "error": f"Unknown agent type: {task.agent_type}"
            }
    
    async def _execute_document_analysis(self, agent: DocumentAnalysisAgent, task: AgentTask) -> Dict[str, Any]:
        """Execute document analysis task."""
        documents = task.input_data.get("documents", [])
        analysis_results = []
        
        for doc in documents:
            result = await agent.analyze_document(
                document_content=doc["content"],
                case_context=asdict(task.case_context)
            )
            analysis_results.append(result)
        
        return {
            "status": "completed",
            "analysis_results": analysis_results,
            "document_count": len(documents),
            "execution_timestamp": datetime.now().isoformat()
        }
    
    async def _execute_compliance_validation(self, agent: ComplianceAgent, task: AgentTask) -> Dict[str, Any]:
        """Execute compliance validation task."""
        result = await agent.validate_compliance(task.input_data)
        
        return {
            "status": "completed",
            "validation_result": result,
            "execution_timestamp": datetime.now().isoformat()
        }
    
    async def _store_task_result(self, task: AgentTask, result: Dict[str, Any]):
        """Store task result in memory."""
        result_content = f"""
        Task Result: {task.task_id}
        Agent: {task.agent_type}
        Status: {result.get('status', 'unknown')}
        Execution Time: {result.get('execution_timestamp', 'unknown')}
        """
        
        if result.get("status") == "completed":
            if task.agent_type == "document_analysis":
                doc_count = result.get("document_count", 0)
                result_content += f"Documents Analyzed: {doc_count}"
            elif task.agent_type == "compliance":
                validation = result.get("validation_result", {})
                result_content += f"Compliance Status: {validation.get('compliance_status', 'unknown')}"
        
        await self.memory_manager.add_case_memory(
            case_id=task.case_context.case_id,
            content=result_content,
            metadata={
                "memory_type": "task_result",
                "task_id": task.task_id,
                "agent_type": task.agent_type,
                "category": "process_workflows",
                "subcategory": "task_execution"
            }
        )
    
    async def _store_processing_summary(self, case_context: CaseContext, results: Dict[str, Any]):
        """Store processing summary in memory."""
        summary_content = f"""
        Case Processing Summary:
        - Case ID: {case_context.case_id}
        - Total Tasks: {len(results)}
        - Completed Tasks: {len([r for r in results.values() if r.get('status') == 'completed'])}
        - Failed Tasks: {len([r for r in results.values() if r.get('status') == 'error'])}
        - Processing Completed: {datetime.now().isoformat()}
        """
        
        await self.memory_manager.add_case_memory(
            case_id=case_context.case_id,
            content=summary_content,
            metadata={
                "memory_type": "processing_summary",
                "category": "process_workflows",
                "subcategory": "case_completion"
            }
        )
    
    def _sort_tasks_by_dependencies(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """Sort tasks based on dependencies and priority."""
        # Simple implementation - in production, use topological sort
        priority_order = {"high": 0, "normal": 1, "low": 2}
        
        # Sort by priority first, then by dependencies
        sorted_tasks = sorted(tasks, key=lambda t: (
            priority_order.get(t.priority, 2),
            len(t.dependencies)
        ))
        
        return sorted_tasks
    
    async def get_case_memory_summary(self, case_id: str) -> Dict[str, Any]:
        """Get comprehensive memory summary for a case."""
        try:
            # Search all case memories
            case_memories = await self.memory_manager.search_case_memories(case_id)
            
            # Organize by category
            organized_memories = {
                "case_context": [],
                "task_results": [],
                "document_analysis": [],
                "compliance_validation": [],
                "processing_summary": [],
                "other": []
            }
            
            for memory in case_memories:
                metadata = memory.get("metadata", {})
                memory_type = metadata.get("memory_type", "other")
                
                if memory_type in organized_memories:
                    organized_memories[memory_type].append(memory)
                else:
                    organized_memories["other"].append(memory)
            
            return {
                "case_id": case_id,
                "total_memories": len(case_memories),
                "organized_memories": organized_memories,
                "retrieval_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "case_id": case_id,
                "error": str(e),
                "total_memories": 0
            }
    
    async def search_similar_cases(self, case_context: CaseContext, limit: int = 5) -> List[Dict]:
        """Search for similar cases based on context."""
        try:
            # Build search query from case context
            search_query = f"{case_context.jurisdiction} {case_context.transaction_type} {case_context.property_address}"
            
            # Search for similar cases
            similar_memories = await self.memory_manager.client.search(
                query=search_query,
                filters={
                    "category": "process_workflows",
                    "memory_type": "case_context"
                },
                top_k=limit
            )
            
            return similar_memories
            
        except Exception as e:
            print(f"Error searching similar cases: {e}")
            return []

# Usage example
async def example_orchestration():
    """Example of using the memory-aware orchestrator."""
    
    # Initialize memory manager
    memory_manager = initialize_from_environment()
    if not memory_manager:
        return
    
    # Initialize memory categories
    await memory_manager.initialize_memory_categories()
    
    # Create orchestrator
    orchestrator = MemoryAwareOrchestrator(memory_manager)
    
    # Example case data
    case_data = {
        "case_id": "case_12345",
        "client_id": "client_67890",
        "property_address": "123 Main Street, Austin, TX",
        "jurisdiction": "texas",
        "transaction_type": "residential_sale",
        "documents": [
            {
                "name": "Purchase Agreement",
                "content": """
                PURCHASE AGREEMENT
                
                This Purchase Agreement is made between John Smith (Buyer) and Jane Doe (Seller) 
                for the property located at 123 Main Street, Austin, TX.
                
                Purchase Price: $450,000
                Closing Date: December 15, 2023
                Earnest Money: $5,000
                """
            }
        ]
    }
    
    # Process the case
    results = await orchestrator.process_conveyancing_case(case_data)
    print(f"Case processing results: {results}")
    
    # Get case memory summary
    memory_summary = await orchestrator.get_case_memory_summary("case_12345")
    print(f"Memory summary: {memory_summary}")
    
    # Search for similar cases
    similar_cases = await orchestrator.search_similar_cases(
        CaseContext(
            case_id="case_67890",
            client_id="client_11111",
            property_address="456 Oak Avenue, Austin, TX",
            jurisdiction="texas",
            transaction_type="residential_sale"
        )
    )
    print(f"Found {len(similar_cases)} similar cases")

if __name__ == "__main__":
    from memory.memory_config import initialize_from_environment
    
    asyncio.run(example_orchestration())

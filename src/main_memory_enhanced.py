# Memory-Enhanced Main Application

"""
Enhanced main application for conveyancing automation with memory integration.
This replaces the original main.py with memory-enhanced capabilities.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import sys
sys.path.append('.')
sys.path.append('./src')

# Memory integration imports
from memory.implementation_patterns.memory_config import initialize_from_environment, ConveyancingMemoryManager
from memory.implementation_patterns.memory_orchestrator import MemoryAwareOrchestrator
from memory.implementation_patterns.memory_enhanced_agents import DocumentAnalysisAgent, ComplianceAgent
from memory.testing_strategies.memory_test_suite import run_comprehensive_tests

# Original imports (preserved for compatibility - commented out as q-and-a-orchestra-agent was removed)
# from q_and_a_orchestra_agent.core.orchestrator import Orchestrator
# from q_and_a_orchestra_agent.core.model_router import ModelRouter
# from q_and_a_orchestra_agent.core.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryEnhancedConveyancingApp:
    """Main application class with memory enhancement."""
    
    def __init__(self):
        """Initialize the memory-enhanced application."""
        self.memory_manager: Optional[ConveyancingMemoryManager] = None
        self.orchestrator: Optional[MemoryAwareOrchestrator] = None
        self.original_orchestrator: Optional[Orchestrator] = None
        self.model_router: Optional[ModelRouter] = None
        self.config: Optional[Config] = None
        
        # Application state
        self.is_initialized = False
        self.memory_enabled = False
        self.active_cases = {}
        
    async def initialize(self) -> bool:
        """
        Initialize the application with memory enhancement.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("🚀 Initializing Memory-Enhanced Conveyancing Application")
            
            # Load configuration
            self.config = Config()
            await self.config.load()
            
            # Initialize memory system
            self.memory_manager = initialize_from_environment()
            if self.memory_manager:
                logger.info("✅ Memory manager initialized")
                self.memory_enabled = True
                
                # Initialize memory categories
                init_results = await self.memory_manager.initialize_memory_categories()
                logger.info(f"✅ Memory categories initialized: {sum(init_results.values())}/{len(init_results)} successful")
                
                # Create memory-enhanced orchestrator
                self.orchestrator = MemoryAwareOrchestrator(self.memory_manager)
                logger.info("✅ Memory-enhanced orchestrator created")
                
            else:
                logger.warning("⚠️ Memory manager not initialized - running without memory enhancement")
                self.memory_enabled = False
            
            # Initialize original components for compatibility
            self.model_router = ModelRouter(self.config)
            self.original_orchestrator = Orchestrator(self.model_router, self.config)
            
            self.is_initialized = True
            logger.info("✅ Application initialization complete")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Application initialization failed: {e}")
            return False
    
    async def process_conveyancing_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a conveyancing request with memory enhancement.
        
        Args:
            request_data: Request data including case information
            
        Returns:
            Processing results
        """
        if not self.is_initialized:
            return {
                "status": "error",
                "error": "Application not initialized"
            }
        
        try:
            logger.info(f"🔄 Processing conveyancing request: {request_data.get('case_id', 'unknown')}")
            
            # Use memory-enhanced processing if available
            if self.memory_enabled and self.orchestrator:
                result = await self.orchestrator.process_conveyancing_case(request_data)
                
                # Add memory summary to result
                memory_summary = await self.orchestrator.get_case_memory_summary(request_data.get('case_id'))
                result["memory_summary"] = memory_summary
                
                logger.info(f"✅ Memory-enhanced processing completed for case {request_data.get('case_id')}")
                return result
            
            else:
                # Fallback to original processing
                result = await self.original_orchestrator.process_request(request_data)
                logger.info(f"✅ Standard processing completed for case {request_data.get('case_id')}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Request processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "case_id": request_data.get('case_id')
            }
    
    async def search_case_memories(self, case_id: str, query: str = "") -> Dict[str, Any]:
        """
        Search memories for a specific case.
        
        Args:
            case_id: Case identifier
            query: Optional search query
            
        Returns:
            Search results
        """
        if not self.memory_enabled or not self.memory_manager:
            return {
                "status": "error",
                "error": "Memory system not available"
            }
        
        try:
            memories = await self.memory_manager.search_case_memories(case_id, query)
            return {
                "status": "success",
                "case_id": case_id,
                "query": query,
                "memories": memories,
                "count": len(memories)
            }
        except Exception as e:
            logger.error(f"❌ Memory search failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "case_id": case_id
            }
    
    async def get_jurisdiction_requirements(self, jurisdiction: str) -> Dict[str, Any]:
        """
        Get jurisdiction-specific requirements.
        
        Args:
            jurisdiction: Jurisdiction name
            
        Returns:
            Jurisdiction requirements
        """
        if not self.memory_enabled or not self.memory_manager:
            return {
                "status": "error",
                "error": "Memory system not available"
            }
        
        try:
            requirements = await self.memory_manager.get_jurisdiction_requirements(jurisdiction)
            return {
                "status": "success",
                "jurisdiction": jurisdiction,
                "requirements": requirements,
                "count": len(requirements)
            }
        except Exception as e:
            logger.error(f"❌ Jurisdiction requirements retrieval failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "jurisdiction": jurisdiction
            }
    
    async def run_system_tests(self) -> Dict[str, Any]:
        """Run comprehensive system tests."""
        logger.info("🧪 Running comprehensive system tests")
        
        try:
            test_results = await run_comprehensive_tests()
            logger.info(f"✅ System tests completed: {test_results['overall_status']}")
            return test_results
        except Exception as e:
            logger.error(f"❌ System tests failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status = {
            "application_initialized": self.is_initialized,
            "memory_enabled": self.memory_enabled,
            "active_cases": len(self.active_cases),
            "timestamp": datetime.now().isoformat()
        }
        
        if self.memory_enabled and self.memory_manager:
            status["memory_system"] = "active"
        else:
            status["memory_system"] = "inactive"
        
        return status
    
    async def shutdown(self):
        """Shutdown the application gracefully."""
        logger.info("🔄 Shutting down Memory-Enhanced Conveyancing Application")
        
        # Save any pending data
        if self.memory_enabled and self.active_cases:
            logger.info(f"💾 Saving {len(self.active_cases)} active cases")
            # Implementation for saving active cases would go here
        
        # Cleanup resources
        if self.original_orchestrator:
            await self.original_orchestrator.cleanup()
        
        logger.info("✅ Application shutdown complete")

# CLI Interface
class ConveyancingCLI:
    """Command-line interface for the conveyancing application."""
    
    def __init__(self):
        self.app = MemoryEnhancedConveyancingApp()
    
    async def run(self, args: List[str]):
        """Run the CLI with given arguments."""
        if not await self.app.initialize():
            print("❌ Failed to initialize application")
            return
        
        if len(args) == 0:
            await self.show_help()
            return
        
        command = args[0].lower()
        
        if command == "process":
            await self.process_request(args[1:])
        elif command == "search":
            await self.search_memories(args[1:])
        elif command == "jurisdiction":
            await self.get_jurisdiction(args[1:])
        elif command == "test":
            await self.run_tests()
        elif command == "status":
            await self.show_status()
        else:
            await self.show_help()
    
    async def process_request(self, args: List[str]):
        """Process a conveyancing request."""
        if len(args) == 0:
            print("❌ Please provide a request file")
            return
        
        request_file = args[0]
        try:
            with open(request_file, 'r') as f:
                request_data = json.load(f)
            
            result = await self.app.process_conveyancing_request(request_data)
            print(f"✅ Request processed: {result['status']}")
            
            if result.get("memory_summary"):
                print(f"📊 Memory Summary: {result['memory_summary']['total_memories']} memories")
            
        except Exception as e:
            print(f"❌ Error processing request: {e}")
    
    async def search_memories(self, args: List[str]):
        """Search case memories."""
        if len(args) < 1:
            print("❌ Please provide a case ID")
            return
        
        case_id = args[0]
        query = " ".join(args[1:]) if len(args) > 1 else ""
        
        result = await self.app.search_case_memories(case_id, query)
        print(f"🔍 Found {result.get('count', 0)} memories for case {case_id}")
        
        for memory in result.get("memories", [])[:5]:  # Show first 5
            print(f"  - {memory.get('memory', 'No content')[:100]}...")
    
    async def get_jurisdiction(self, args: List[str]):
        """Get jurisdiction requirements."""
        if len(args) == 0:
            print("❌ Please provide a jurisdiction")
            return
        
        jurisdiction = args[0]
        result = await self.app.get_jurisdiction_requirements(jurisdiction)
        print(f"📋 Found {result.get('count', 0)} requirements for {jurisdiction}")
        
        for requirement in result.get("requirements", [])[:3]:  # Show first 3
            print(f"  - {requirement.get('memory', 'No content')[:100]}...")
    
    async def run_tests(self):
        """Run system tests."""
        result = await self.app.run_system_tests()
        print(f"🧪 Test Results: {result['overall_status']}")
        
        if result.get("unit_tests"):
            unit = result["unit_tests"]["test_summary"]
            print(f"  Unit Tests: {unit['passed']}/{unit['total']} ({unit['success_rate']:.1f}%)")
        
        if result.get("integration_tests"):
            integration = result["integration_tests"]
            print(f"  Integration Tests: {integration['overall_status']}")
    
    async def show_status(self):
        """Show system status."""
        status = await self.app.get_system_status()
        print(f"📊 System Status:")
        print(f"  Initialized: {status['application_initialized']}")
        print(f"  Memory Enabled: {status['memory_enabled']}")
        print(f"  Memory System: {status['memory_system']}")
        print(f"  Active Cases: {status['active_cases']}")
        print(f"  Timestamp: {status['timestamp']}")
    
    async def show_help(self):
        """Show help information."""
        print("""
🚀 Memory-Enhanced Conveyancing Automation System

Commands:
  process <file>     - Process a conveyancing request from JSON file
  search <case_id> [query] - Search memories for a case
  jurisdiction <name> - Get jurisdiction-specific requirements
  test             - Run comprehensive system tests
  status           - Show system status
  help             - Show this help message

Examples:
  python main_memory_enhanced.py process request.json
  python main_memory_enhanced.py search case_12345 "client request"
  python main_memory_enhanced.py jurisdiction texas
  python main_memory_enhanced.py test
        """)

# Main execution
async def main():
    """Main execution function."""
    import sys
    
    app = MemoryEnhancedConveyancingApp()
    
    try:
        # Initialize application
        if not await app.initialize():
            print("❌ Failed to initialize application")
            sys.exit(1)
        
        # Check if running as CLI or server
        if len(sys.argv) > 1:
            # CLI mode
            cli = ConveyancingCLI()
            await cli.run(sys.argv[1:])
        else:
            # Server mode - would start web server here
            print("🌐 Starting web server mode...")
            print("📊 System Status:", await app.get_system_status())
            print("💡 Use 'python main_memory_enhanced.py help' for CLI commands")
            
            # Keep running
            try:
                while True:
                    await asyncio.sleep(60)
                    # Periodic status check could go here
            except KeyboardInterrupt:
                print("\n🛑 Received interrupt signal")
    
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        sys.exit(1)
    finally:
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

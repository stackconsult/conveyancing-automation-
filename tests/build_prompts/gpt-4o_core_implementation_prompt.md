# Conveyancing Automation Core Implementation

## Model: gpt-4o
## Phase: core_implementation
## Priority: 5/5
## Estimated Tokens: 12000

## Objective
Implement production-grade Stage 1 Retrieval System with comprehensive testing

## Success Metrics
- Complete implementation of all retrieval components
- Comprehensive test suite with 95%+ coverage
- Production-ready API with full documentation
- Database implementation with migrations
- Vector database and Mem0 integration
- Docker configuration and deployment setup

## Validation Steps
- Code review for quality and standards compliance
- Test execution for coverage and functionality
- API testing for endpoints and validation
- Database migration testing
- Performance benchmarking
- Security vulnerability scanning

## Output Format
Complete Python codebase with tests, documentation, and deployment configuration

---

## SYSTEM PROMPT
```

# SYSTEM: PRODUCTION-GRADE SOFTWARE ENGINEER

You are a **Senior Software Engineer** with 10+ years of experience building production-grade systems in Python, FastAPI, and modern cloud architectures. You specialize in:

- **Clean Code Architecture** with SOLID principles and design patterns
- **Type Safety** with Pydantic models and comprehensive validation
- **Async/Python Programming** with asyncio and performance optimization
- **Database Design** with SQLAlchemy ORM and query optimization
- **API Development** with FastAPI, OpenAPI, and RESTful patterns
- **Testing Excellence** with pytest, mocking, and comprehensive coverage
- **Production Deployment** with Docker, CI/CD, and monitoring

## ENGINEERING PRINCIPLES

1. **Code Quality First**: Every line must be readable, maintainable, and tested
2. **Type Safety**: All interfaces must be strongly typed with no runtime surprises
3. **Performance by Design**: Efficient algorithms, database queries, and memory usage
4. **Error Handling**: Comprehensive error handling with proper logging and recovery
5. **Security by Default**: Input validation, SQL injection prevention, and data protection
6. **Testability**: Every component must be unit testable with dependency injection

## TECHNICAL STANDARDS

- **Python 3.11+** with modern type hints and async/await
- **FastAPI** with dependency injection and automatic validation
- **SQLAlchemy 2.0** with async sessions and relationship loading
- **Pydantic V2** for data validation and serialization
- **pytest** with async testing and comprehensive fixtures
- **Docker** with multi-stage builds and security scanning
- **GitHub Actions** for CI/CD with automated testing and deployment

## CODE QUALITY REQUIREMENTS

- **95%+ test coverage** with meaningful tests
- **Type hints** on all functions and classes
- **Comprehensive error handling** with custom exceptions
- **Structured logging** with correlation IDs
- **Performance monitoring** with metrics and tracing
- **Security scanning** with zero vulnerabilities
- **Documentation** with docstrings and examples

## OUTPUT REQUIREMENTS

Your implementation must include:
1. **Complete source code** with production-grade quality
2. **Comprehensive tests** with high coverage and edge cases
3. **Database models** with proper relationships and migrations
4. **API endpoints** with validation and error handling
5. **Configuration management** with environment-specific settings
6. **Monitoring and logging** with structured output
7. **Docker configuration** with security best practices
8. **Documentation** with usage examples and API specs

You must provide **production-ready** code that can be immediately deployed without additional development.

```

---

## USER PROMPT
```

# CORE IMPLEMENTATION REQUEST

## PROJECT OVERVIEW
**Project**: Conveyancing Automation System
**Repository**: https://github.com/stackconsult/conveyancing-automation-
**Phase**: core_implementation
**Architecture Status**: Complete architecture design available

## CURRENT SYSTEM STATE
{
  "stage1_retrieval": "Complete implementation with schemas, algorithms, and integration",
  "mem0_integration": "4-layer memory architecture ready",
  "vector_database": "Needs implementation for semantic search",
  "deepseek_r1": "Ready for integration with context packages",
  "langgraph": "Integration framework ready",
  "testing": "Comprehensive test suite implemented",
  "documentation": "Complete technical specification available"
}

## IMPLEMENTATION REQUIREMENTS

### 1. Core Retrieval System Implementation
Implement the complete Stage 1 Retrieval System based on the provided architecture:

**Components to Implement:**
- **RetrievalAgent** orchestrator with comprehensive error handling
- **SegmentAwareRetriever** for hybrid semantic + structural search
- **RiskAwareRanker** for risk-weighted scoring without tradeoffs
- **ContextPackager** for bounded DeepSeek-R1 context packages
- **CoverageSelfCheck** for automatic gap detection and patching

**Technical Requirements:**
- Async/await patterns throughout with proper concurrency
- Type safety with Pydantic models and validation
- Comprehensive error handling with custom exceptions
- Structured logging with correlation IDs
- Performance monitoring with metrics collection
- Unit tests with 95%+ coverage

### 2. Database Implementation
Implement complete data layer:

**Models to Create:**
- **DocumentChunk** model with metadata and indexing
- **RetrievalIntent** model with query parameters
- **ContextPackage** model with bounded context
- **RetrievalSummary** model with audit information
- **AuditEntry** model for compliance tracking

**Database Requirements:**
- SQLAlchemy 2.0 with async sessions
- Proper relationships and foreign keys
- Database migrations with Alembic
- Query optimization with indexes
- Connection pooling and retry logic

### 3. API Implementation
Implement comprehensive REST API:

**Endpoints to Create:**
- **POST /retrieval/search** - Execute retrieval with intent
- **GET /retrieval/status/{retrieval_id}** - Check retrieval status
- **GET /retrieval/results/{retrieval_id}** - Get retrieval results
- **POST /chunks** - Register new document chunks
- **GET /chunks/{chunk_id}** - Retrieve chunk metadata
- **GET /health** - System health check

**API Requirements:**
- FastAPI with dependency injection
- Request/response validation with Pydantic
- Comprehensive error handling with HTTP status codes
- Rate limiting and authentication middleware
- OpenAPI documentation with examples
- API versioning and backward compatibility

### 4. Vector Database Integration
Implement vector search capabilities:

**Integration Requirements:**
- Qdrant client with connection management
- Vector embedding with OpenAI/Local models
- Semantic search with filtering capabilities
- Index management and optimization
- Batch processing for performance

### 5. Mem0 Integration
Implement 4-layer memory architecture:

**Memory Layers:**
- **Deal Memory**: Per-deal conversation and context
- **Agent Memory**: Per-agent operational patterns
- **Global Memory**: Cross-deal patterns and knowledge
- **Meta Memory**: System learning and optimization

**Integration Requirements:**
- Mem0 client with proper error handling
- Memory scoping and isolation
- Pattern mining and retrieval
- Memory cleanup and retention policies

### 6. Testing Implementation
Create comprehensive test suite:

**Test Categories:**
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Performance Tests**: Load and stress testing
- **Security Tests**: Vulnerability and penetration testing

**Testing Requirements:**
- pytest with async testing support
- Mocking for external dependencies
- Test fixtures and data factories
- Coverage reporting with 95%+ threshold
- Performance benchmarking

## CONSTRAINTS AND DEPENDENCIES

### Technical Constraints:
- Production-grade quality with 95%+ test coverage
- Alberta conveyancing law compliance
- Sub-100ms API response times
- Zero security vulnerabilities
- Scalable to 1000+ concurrent deals
- Complete audit trail for compliance

### System Dependencies:
- Python 3.11+
- FastAPI
- SQLAlchemy 2.0
- Pydantic V2
- Mem0 Platform
- Vector Database (Qdrant/Weaviate)
- LangGraph
- DeepSeek-R1
- Docker
- PostgreSQL

### Success Criteria:
- Complete system architecture with all components
- Production-ready implementation with comprehensive testing
- Alberta-specific domain logic with legal accuracy
- End-to-end integration with all systems
- Performance benchmarks meeting requirements
- Security and compliance validation

## DELIVERABLES REQUIRED

1. **Complete Source Code**
   - All components implemented with production quality
   - Type safety with comprehensive validation
   - Error handling and logging throughout
   - Performance optimization and monitoring

2. **Database Implementation**
   - SQLAlchemy models with relationships
   - Database migrations with Alembic
   - Query optimization and indexing
   - Connection management and pooling

3. **API Implementation**
   - FastAPI endpoints with validation
   - OpenAPI documentation with examples
   - Authentication and authorization
   - Rate limiting and security middleware

4. **Testing Suite**
   - Unit tests with 95%+ coverage
   - Integration tests for component interactions
   - Performance tests with benchmarks
   - Security tests with vulnerability scanning

5. **Configuration and Deployment**
   - Docker configuration with multi-stage builds
   - Environment-specific configuration
   - CI/CD pipeline with GitHub Actions
   - Monitoring and logging setup

## QUALITY REQUIREMENTS

- **Production-Ready**: Code must be deployable without additional development
- **Type Safe**: Comprehensive type hints with no runtime errors
- **Well-Tested**: 95%+ coverage with meaningful tests
- **Performant**: Optimized for speed and memory usage
- **Secure**: No vulnerabilities with proper validation
- **Maintainable**: Clean code with documentation

Please provide the complete implementation with production-grade quality ready for immediate deployment.

```

---

## Context Information
```json
{
  "project_name": "Conveyancing Automation System",
  "repository_url": "https://github.com/stackconsult/conveyancing-automation-",
  "target_phase": "core_implementation",
  "dependencies": [
    "Python 3.11+",
    "FastAPI",
    "SQLAlchemy 2.0",
    "Pydantic V2",
    "Mem0 Platform",
    "Vector Database (Qdrant/Weaviate)",
    "LangGraph",
    "DeepSeek-R1",
    "Docker",
    "PostgreSQL"
  ],
  "constraints": [
    "Production-grade quality with 95%+ test coverage",
    "Alberta conveyancing law compliance",
    "Sub-100ms API response times",
    "Zero security vulnerabilities",
    "Scalable to 1000+ concurrent deals",
    "Complete audit trail for compliance"
  ],
  "success_criteria": [
    "Complete system architecture with all components",
    "Production-ready implementation with comprehensive testing",
    "Alberta-specific domain logic with legal accuracy",
    "End-to-end integration with all systems",
    "Performance benchmarks meeting requirements",
    "Security and compliance validation"
  ]
}
```

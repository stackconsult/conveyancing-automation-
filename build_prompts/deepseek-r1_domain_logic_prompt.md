# Alberta Conveyancing Domain Logic

## Model: deepseek-r1
## Phase: domain_logic
## Priority: 5/5
## Estimated Tokens: 10000

## Objective
Implement specialized legal domain expertise for Alberta conveyancing compliance and risk assessment

## Success Metrics
- Complete Alberta conveyancing domain knowledge base
- Advanced document analysis with semantic understanding
- Comprehensive risk assessment framework
- Regulatory compliance validation engine
- Multi-pass reasoning system with report generation
- Professional-grade legal recommendations

## Validation Steps
- Legal accuracy validation against Alberta regulations
- Risk assessment accuracy testing with known cases
- Compliance validation against regulatory requirements
- Document analysis accuracy with sample documents
- Report quality assessment with legal professionals
- End-to-end testing with complete conveyancing scenarios

## Output Format
Complete domain logic implementation with legal knowledge base, analysis engines, and reporting systems

---

## SYSTEM PROMPT
```

# SYSTEM: LEGAL DOMAIN EXPERT & COMPLIANCE SPECIALIST

You are a **Senior Legal Technology Specialist** with 15+ years of experience in Alberta conveyancing law, document analysis, and regulatory compliance. You specialize in:

- **Alberta Conveyancing Law** with deep knowledge of the Land Titles Act, Torrens system, and real property regulations
- **Legal Document Analysis** with expertise in title searches, mortgages, caveats, and condominium governance
- **Risk Assessment** with focus on title defects, encumbrances, and compliance violations
- **Regulatory Compliance** with Alberta Land Titles Office requirements and legal practice standards
- **Document Intelligence** with advanced pattern recognition and semantic understanding
- **Legal Reasoning** with logical deduction and precedent-based analysis

## DOMAIN EXPERTISE AREAS

### Alberta Conveyancing Law
- **Land Titles System**: Torrens registration, indefeasibility of title, and registration priorities
- **Title Searches**: Instrument registers, caveats, liens, and encumbrances
- **Mortgage Law**: Registration, priority, discharge, and enforcement
- **Condominium Law**: Bylaws, reserve funds, special assessments, and governance
- **Tax Compliance**: Property tax arrears, municipal charges, and tax certificates

### Document Analysis
- **Legal Document Structure**: Understanding legal terminology, clauses, and provisions
- **Risk Pattern Recognition**: Identifying common title defects and compliance issues
- **Semantic Understanding**: Contextual analysis of legal language and implications
- **Cross-Reference Analysis**: Connecting related documents and provisions

### Compliance Framework
- **Alberta Land Titles Office**: Registration requirements and procedures
- **Legal Practice Standards**: Professional responsibility and client service standards
- **Regulatory Requirements**: Compliance with provincial and federal regulations
- **Risk Management**: Identification and mitigation of legal risks

## REASONING CAPABILITIES

1. **Multi-Pass Analysis**: Deep examination of documents from multiple perspectives
2. **Context Synthesis**: Combining information from multiple sources and documents
3. **Risk Assessment**: Evaluating potential legal issues and their implications
4. **Compliance Validation**: Ensuring adherence to legal and regulatory requirements
5. **Decision Logic**: Making informed recommendations based on legal analysis

## OUTPUT REQUIREMENTS

Your domain logic implementation must include:
1. **Legal Document Analysis** with Alberta-specific understanding
2. **Risk Assessment Logic** with comprehensive evaluation criteria
3. **Compliance Validation** with regulatory requirement checking
4. **Reasoning Engine** with multi-pass analysis capabilities
5. **Report Generation** with clear, actionable recommendations
6. **Pattern Recognition** for common legal issues and risks
7. **Context Integration** with memory and historical patterns

You must provide **legally sound** domain logic that can be trusted for professional conveyancing services.

```

---

## USER PROMPT
```

# DOMAIN LOGIC IMPLEMENTATION REQUEST

## PROJECT OVERVIEW
**Project**: Conveyancing Automation System
**Repository**: https://github.com/stackconsult/conveyancing-automation-
**Phase**: domain_logic
**Domain**: Alberta Conveyancing Law and Compliance

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

## DOMAIN LOGIC REQUIREMENTS

### 1. Alberta Conveyancing Expertise
Implement specialized knowledge of Alberta conveyancing law:

**Legal Areas to Cover:**
- **Land Titles System**: Torrens registration, indefeasibility, registration priorities
- **Title Search Analysis**: Instruments, caveats, liens, mortgages, and encumbrances
- **Mortgage Law**: Registration, priority, discharge, enforcement procedures
- **Condominium Governance**: Bylaws, reserve funds, special assessments, meetings
- **Tax Compliance**: Property taxes, municipal charges, tax certificates
- **Transfer of Land**: Registration requirements, execution, and delivery

**Implementation Requirements:**
- Legal terminology and concept definitions
- Regulatory requirement mappings
- Compliance rule implementations
- Risk factor identification and scoring
- Legal precedent and pattern recognition

### 2. Document Analysis Engine
Implement advanced document analysis capabilities:

**Analysis Components:**
- **Semantic Understanding**: Contextual analysis of legal language
- **Structure Recognition**: Identifying document sections and provisions
- **Risk Pattern Detection**: Finding common title defects and issues
- **Cross-Reference Analysis**: Connecting related information across documents
- **Temporal Analysis**: Understanding dates, deadlines, and time-sensitive requirements

**Technical Requirements:**
- Natural language processing for legal text
- Pattern matching for common legal phrases
- Context extraction and classification
- Risk scoring algorithms
- Compliance validation logic

### 3. Risk Assessment Framework
Implement comprehensive risk evaluation:

**Risk Categories:**
- **Title Risks**: Undischarged mortgages, caveats, liens, ownership issues
- **Compliance Risks**: Missing documents, improper execution, regulatory violations
- **Financial Risks**: Tax arrears, outstanding charges, assessment deficiencies
- **Governance Risks**: Condo bylaw violations, special assessments, reserve issues
- **Procedural Risks**: Missing signatures, improper registration, timing issues

**Assessment Requirements:**
- Risk scoring algorithms with weighted factors
- Risk severity classification (Critical, High, Medium, Low)
- Risk mitigation recommendations
- Historical pattern analysis
- Client communication guidelines

### 4. Compliance Validation Engine
Implement regulatory compliance checking:

**Compliance Areas:**
- **Alberta Land Titles Office**: Registration requirements and procedures
- **Legal Practice Standards**: Professional responsibility and client service
- **Regulatory Requirements**: Provincial and federal compliance
- **Document Standards**: Formatting, execution, and content requirements

**Validation Requirements:**
- Compliance rule engine with configurable criteria
- Automated compliance checking with detailed reports
- Exception handling for special circumstances
- Regulatory update monitoring and implementation
- Audit trail generation for compliance verification

### 5. Reasoning Engine
Implement multi-pass analysis capabilities:

**Reasoning Passes:**
- **Pass 1**: Document structure and content analysis
- **Pass 2**: Risk identification and assessment
- **Pass 3**: Compliance validation and checking
- **Pass 4**: Context synthesis and pattern recognition
- **Pass 5**: Recommendation generation and prioritization

**Engine Requirements:**
- Sequential analysis with information passing between passes
- Confidence scoring for each analysis result
- Contradiction detection and resolution
- Evidence gathering and citation
- Explainable reasoning with audit trails

### 6. Report Generation System
Implement comprehensive reporting capabilities:

**Report Types:**
- **Title Risk Report**: Detailed analysis of title issues and risks
- **Compliance Report**: Regulatory compliance status and violations
- **Recommendation Report**: Actionable recommendations with priorities
- **Summary Report**: Executive summary with key findings
- **Audit Report**: Detailed audit trail and analysis documentation

**Generation Requirements:**
- Template-based report generation with customization
- Clear, actionable language for legal professionals
- Evidence-based recommendations with citations
- Risk prioritization with mitigation strategies
- Professional formatting and presentation

## CONSTRAINTS AND DEPENDENCIES

### Legal Constraints:
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

1. **Legal Domain Knowledge Base**
   - Alberta conveyancing law concepts and definitions
   - Regulatory requirements and compliance rules
   - Risk factors and assessment criteria
   - Legal precedents and pattern recognition

2. **Document Analysis Engine**
   - Semantic understanding of legal documents
   - Structure recognition and content extraction
   - Risk pattern detection and classification
   - Cross-reference analysis and context integration

3. **Risk Assessment Framework**
   - Comprehensive risk scoring algorithms
   - Risk severity classification system
   - Mitigation recommendation engine
   - Historical pattern analysis capabilities

4. **Compliance Validation System**
   - Regulatory compliance rule engine
   - Automated compliance checking with reporting
   - Exception handling and special circumstances
   - Audit trail generation and documentation

5. **Reasoning and Reporting Engine**
   - Multi-pass analysis with information synthesis
   - Comprehensive report generation with templates
   - Evidence-based recommendations with citations
   - Professional formatting and presentation

## QUALITY REQUIREMENTS

- **Legally Sound**: All logic must comply with Alberta law and regulations
- **Comprehensive**: Cover all aspects of conveyancing law and compliance
- **Accurate**: Risk assessment and compliance validation must be reliable
- **Explainable**: All reasoning must be traceable and justifiable
- **Actionable**: Recommendations must be clear and implementable
- **Professional**: Output must meet legal practice standards

Please provide the complete domain logic implementation with professional-grade legal expertise.

```

---

## Context Information
```json
{
  "project_name": "Conveyancing Automation System",
  "repository_url": "https://github.com/stackconsult/conveyancing-automation-",
  "target_phase": "domain_logic",
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

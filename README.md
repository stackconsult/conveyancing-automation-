# Conveyancing Automation System

**Memory-enhanced legal conveyance automation with intelligent document processing and compliance validation.**

A production-grade system that transforms real estate legal conveyancing through persistent memory integration, intelligent agent orchestration, and automated compliance checking.

---

## 🚀 Quick Start

### Prerequisites

1. **Mem0 Platform Account**: Get API key from [Mem0 Dashboard](https://app.mem0.ai/dashboard/api-keys)
2. **Python Environment**: Python 3.11+
3. **Docker & Docker Compose**: For containerized deployment

### Installation

1. **Clone Repository**:

   ```bash
   git clone https://github.com/stackconsult/conveyancing-automation-system.git
   cd conveyancing-automation-system
   ```

2. **Environment Setup**:

   ```bash
   cp .env.memory_enhanced.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Docker Deployment** (Recommended):

   ```bash
   docker-compose -f docker-compose.memory_enhanced.yml up -d
   ```

4. **Local Development**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements_memory_enhanced.txt
   python main_memory_enhanced.py
   ```

---

## 🎯 What This Does

### Core Capabilities

#### 1️⃣ Memory-Enhanced Document Processing

Automatically analyzes legal documents with contextual memory:

- **Document Type Recognition** - Purchase agreements, deeds, disclosures
- **Key Term Extraction** - Parties, prices, dates, contingencies
- **Legal Precedent Integration** - Relevant case law and statutes
- **Compliance Checking** - Real-time regulatory validation

#### 2️⃣ Intelligent Compliance Validation

Automated compliance checking with jurisdiction-specific rules:

- **Multi-Jurisdiction Support** - Texas, California, New York, Florida, Illinois
- **Regulatory Requirements** - State-specific disclosure and filing rules
- **Risk Assessment** - Legal and financial risk identification
- **Audit Trail Generation** - Complete compliance documentation

#### 3️⃣ Case Management with Memory

Persistent memory for complete case lifecycle:

- **Client Preference Tracking** - Communication and scheduling preferences
- **Stakeholder Coordination** - Agent, lender, title company interactions
- **Process Workflow Management** - Step-by-step conveyancing procedures
- **Historical Context** - Previous cases and precedent analysis

---

## 📖 How to Use

### Basic Pattern

1. **Submit Case Data** - Property details, parties, jurisdiction
2. **Upload Documents** - Purchase agreements, disclosures, title reports
3. **Automatic Processing** - Document analysis and compliance validation
4. **Review Results** - Memory-enhanced insights and recommendations
5. **Complete Transaction** - Coordinated closing with all stakeholders

### Example Usage

```python
# Process a conveyancing case
case_data = {
    "case_id": "case_12345",
    "client_id": "client_67890",
    "property_address": "123 Main Street, Austin, TX",
    "jurisdiction": "texas",
    "transaction_type": "residential_sale",
    "documents": [
        {
            "name": "Purchase Agreement",
            "content": "Purchase agreement content..."
        }
    ]
}

# Process with memory enhancement
results = await orchestrator.process_conveyancing_case(case_data)
```

### CLI Commands

```bash
# Process a case
python main_memory_enhanced.py process request.json

# Search case memories
python main_memory_enhanced.py search case_12345 "client request"

# Get jurisdiction requirements
python main_memory_enhanced.py jurisdiction texas

# Run system tests
python main_memory_enhanced.py test

# Check system status
python main_memory_enhanced.py status
```

---

## 🏗️ Architecture

### Memory Categories

```
memory/
├── legal_knowledge/     # Property laws, regulations, precedents
├── document_templates/  # Legal contracts, forms, standard documents
├── process_workflows/   # Conveyancing procedures, compliance checklists
├── stakeholder_data/    # Client preferences, agent coordination
├── risk_factors/        # Legal/financial risks, mitigation strategies
├── compliance_rules/    # Regulatory requirements, validation rules
└── market_data/         # Property values, trends, analytics
```

### Agent System

- **Document Analysis Agent** - Memory-enhanced document processing
- **Compliance Agent** - Regulatory validation with context
- **Coordination Agent** - Multi-party workflow management
- **Risk Assessment Agent** - Legal/financial risk evaluation
- **Communication Agent** - Stakeholder notification system

### Memory Integration

- **Contextual Memory Loading** - Relevant memories based on case context
- **Persistent Storage** - All interactions stored for future reference
- **Intelligent Search** - Contextual memory retrieval with filtering
- **Version Control** - Complete audit trail of all changes

---

## 🛠 Technology Stack

### Core Technologies

- **Backend**: Python 3.11+, FastAPI, asyncio
- **Memory**: Mem0 Platform with persistent storage
- **Database**: PostgreSQL with Redis caching
- **Containerization**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana, structured logging

### AI/LLM Integration

- **Local Models**: Ollama (Llama, Qwen, Mistral)
- **Cloud Models**: OpenAI GPT, Anthropic Claude
- **Model Routing**: Intelligent local-first selection
- **Cost Management**: Budget tracking and optimization

### Document Processing

- **PDF Processing**: PyPDF2, pdfplumber
- **Document Parsing**: python-docx, pytesseract
- **Text Analysis**: spaCy, NLTK, scikit-learn
- **Template Matching**: Custom legal document patterns

---

## 📋 Supported Jurisdictions

### Currently Supported

- **Texas** - Complete residential and commercial conveyancing
- **California** - Residential property transactions
- **New York** - Real property transfer procedures
- **Florida** - Residential conveyancing workflows
- **Illinois** - Property transfer compliance

### Adding New Jurisdictions

1. **Create Memory Category**:

   ```python
   await memory_manager.add_case_memory(
       case_id="jurisdiction_setup",
       content="Jurisdiction-specific rules and requirements",
       metadata={"jurisdiction": "new_state", "category": "compliance_rules"}
   )
   ```

2. **Define Compliance Rules**:

   ```python
   compliance_rules = {
       "disclosure_requirements": [...],
       "filing_procedures": [...],
       "timeline_requirements": [...]
   }
   ```

---

## 🔧 Configuration

### Environment Variables

```bash
# Memory System
MEM0_API_KEY=your_mem0_api_key
CONVEYANCING_PROJECT_ID=conveyancing_automation

# LLM Providers
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
OLLAMA_BASE_URL=http://localhost:11434

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/conveyancing
REDIS_URL=redis://localhost:6379/0

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
PORT=8000
```

### Memory Configuration

```python
# Memory categories setup
memory_config = {
    "legal_knowledge_categories": [
        "property_laws", "contract_law", "compliance_frameworks"
    ],
    "document_template_categories": [
        "purchase_agreements", "deed_templates", "disclosure_forms"
    ],
    "compliance_categories": [
        "regulatory_requirements", "validation_rules", "audit_requirements"
    ]
}
```

---

## � Monitoring & Observability

### Health Checks

```bash
# System health
curl http://localhost:8000/health

# Memory system status
curl http://localhost:8000/api/memory/status

# Agent performance
curl http://localhost:8000/api/agents/metrics
```

### Metrics Available

- **Memory Operations** - Search, add, update performance
- **Document Processing** - Analysis time and accuracy
- **Compliance Validation** - Check results and processing time
- **Case Processing** - End-to-end completion rates
- **System Performance** - Resource utilization and error rates

### Monitoring Stack

- **Prometheus** - Metrics collection and storage
- **Grafana** - Visualization and alerting
- **Structured Logging** - JSON logs with correlation IDs
- **Health Checks** - Service availability monitoring

---

## 🧪 Testing

### Run Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Memory system tests
python memory/testing_strategies/memory_test_suite.py

# Performance tests
pytest tests/performance/
```

### Test Coverage

- **Memory Integration** - 95%+ coverage target
- **Agent Functionality** - Document analysis and compliance validation
- **API Endpoints** - REST API and WebSocket connections
- **Error Handling** - Exception scenarios and recovery
- **Performance** - Load testing and optimization validation

---

## � Deployment

### Production Deployment

```bash
# Deploy with Docker Compose
docker-compose -f docker-compose.memory_enhanced.yml up -d

# Scale services
docker-compose -f docker-compose.memory_enhanced.yml up -d --scale conveyancing-app=3

# Monitor deployment
docker-compose -f docker-compose.memory_enhanced.yml logs -f
```

### Environment Setup

1. **Production Environment**:

   ```bash
   ENVIRONMENT=production
   DEBUG=false
   LOG_LEVEL=INFO
   ```

2. **Security Configuration**:

   ```bash
   SECRET_KEY=your_production_secret
   JWT_SECRET_KEY=your_jwt_secret
   SSL_CERT_PATH=/path/to/cert.pem
   ```

3. **Database Setup**:

   ```bash
   POSTGRES_PASSWORD=secure_password
   DATABASE_URL=postgresql://conveyancing:password@postgres:5432/conveyancing
   ```

---

## 🔒 Security & Compliance

### Data Protection

- **Client Confidentiality** - Encrypted memory storage
- **Audit Trails** - Complete transaction logging
- **Access Controls** - Role-based permissions
- **Data Retention** - Configurable retention policies

### Legal Compliance

- **Regulatory Requirements** - Built-in compliance checking
- **Jurisdiction Rules** - Location-specific enforcement
- **Document Security** - Encrypted storage and transmission
- **Privacy Standards** - GDPR/CCPA ready implementation

### Security Features

- **Authentication** - JWT-based user authentication
- **Authorization** - Role-based access control
- **Encryption** - Data at rest and in transit
- **Audit Logging** - Comprehensive security event tracking

---

## � API Documentation

### Core Endpoints

```bash
# Process conveyancing case
POST /api/cases/process
Content-Type: application/json

# Search case memories
GET /api/memories/search/{case_id}?query=...

# Get jurisdiction requirements
GET /api/jurisdictions/{jurisdiction}/requirements

# System health check
GET /api/health
```

### Memory API Integration

```python
# Add case memory
memory_client.add(
    user_id="case_12345",
    messages=[{"role": "system", "content": "Memory content"}],
    metadata={"category": "legal_knowledge", "jurisdiction": "texas"}
)

# Search memories
memory_client.search(
    query="purchase agreement texas",
    filters={"category": "document_templates", "jurisdiction": "texas"}
)
```

---

## 🆘 Troubleshooting

### Common Issues

**Memory System Not Connecting**

- Check MEM0_API_KEY configuration
- Verify network connectivity to api.mem0.ai
- Review authentication token format

**Document Processing Errors**

- Ensure document formats are supported (PDF, DOC, DOCX)
- Check file size limits (max 50MB)
- Verify OCR dependencies for scanned documents

**Compliance Validation Failures**

- Confirm jurisdiction is supported
- Check regulatory data updates
- Review case context completeness

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG python main_memory_enhanced.py

# Memory system diagnostics
python main_memory_enhanced.py status --verbose

# Test memory connectivity
python -c "from memory.memory_config import initialize_from_environment; print('OK')"
```

---

## 🤝 Contributing

### Development Setup

1. **Fork Repository**
2. **Create Feature Branch**: `git checkout -b feature/new-feature`
3. **Install Dependencies**: `pip install -r requirements_memory_enhanced.txt`
4. **Run Tests**: `pytest`
5. **Submit Pull Request**

### Code Standards

- **Python**: Follow PEP 8, use type hints
- **Documentation**: Comprehensive docstrings
- **Testing**: 95%+ coverage requirement
- **Security**: No hardcoded credentials

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🎉 Ready to Use

The conveyancing automation system is now ready for production deployment with:

- ✅ **Memory-Enhanced Processing** - Intelligent document analysis
- ✅ **Automated Compliance** - Real-time regulatory validation  
- ✅ **Multi-Jurisdiction Support** - State-specific legal rules
- ✅ **Production Deployment** - Docker containerization
- ✅ **Enterprise Security** - Data protection and audit trails
- ✅ **Comprehensive Testing** - Unit, integration, and performance tests

**Start automating your conveyancing workflows today! 🚀**

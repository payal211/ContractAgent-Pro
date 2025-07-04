# Legal Contract Analyzer
🏛️ **AI-Powered Legal Contract Analysis System**

A comprehensive agentic AI system built with LangChain, RAG (Retrieval Augmented Generation), local Llama models, and LangGraph for automated legal contract analysis, risk assessment, and compliance checking.

## 🌟 Features

### Core Analysis Capabilities
- **Contract Type Detection**: Automatically identifies contract types (Service Agreement, Employment Contract, NDA, etc.)
- **Entity Extraction**: Extracts key parties, dates, financial terms, and important clauses
- **Risk Assessment**: Identifies potential legal, financial, and operational risks
- **Compliance Checking**: Verifies contracts against legal and regulatory requirements
- **Recommendation Engine**: Provides actionable suggestions for contract improvement

### Advanced Features
- **Multi-format Support**: Processes PDF, TXT, and DOCX files
- **Real-time Processing**: Live status updates during analysis
- **Historical Analysis**: Track and compare multiple contract analyses
- **Risk Dashboard**: Visual analytics for risk trends and patterns
- **RAG-Enhanced Analysis**: Leverages legal knowledge base for informed decisions

## 🏗️ Architecture

### Technology Stack
- **LangChain**: Core AI pipeline and document processing
- **LangGraph**: Workflow orchestration and state management
- **Ollama**: Local LLM deployment (Llama 3.1 8B)
- **ChromaDB**: Vector database for RAG implementation
- **FastAPI**: Backend API service
- **Streamlit**: Interactive web interface
- **PyPDF**: PDF document processing
- **Plotly**: Data visualization and analytics

### System Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   FastAPI       │────│   LangGraph     │
│   (Frontend)    │    │   (Backend)     │    │   (Workflow)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Upload   │    │   Contract      │    │   Entity        │
│   & Processing  │    │   Analysis API  │    │   Extraction    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                │                       ▼
                                │              ┌─────────────────┐
                                │              │   Risk          │
                                │              │   Assessment    │
                                │              └─────────────────┘
                                │                       │
                                │                       ▼
                                │              ┌─────────────────┐
                                │              │   Compliance    │
                                │              │   Check         │
                                │              └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   ChromaDB      │    │   Ollama LLM    │
                       │   (Vector DB)   │    │   (Local)       │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed locally
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/payal211/ContractAgent-Pro.git
cd ContractAgent-Pro
```
2. **Create conda environment**
```bash
conda create -n contract-analyzer python=3.9
conda activate contract-analyzer
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

#### Option 2: Using Virtual Environment

1. **Clone the repository**
```bash
git clone https://github.com/payal211/ContractAgent-Pro.git
cd ContractAgent-Pro
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install and setup Ollama**
```bash
# Install Ollama (visit https://ollama.ai for installation instructions)
# Pull the required model
ollama pull llama3.1:8b
```

5. **Initialize the knowledge base**
```bash
python main.py  # This will create the legal knowledge base
```

### Running the System

1. **Start the FastAPI backend**
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
or 
```bash
python api.py
```

2. **Launch the Streamlit UI**
```bash
streamlit run st_app.py
```

3. **Access the application**
- Open your browser to `http://localhost:8501`
- The API docs are available at `http://localhost:8000/docs`

## 📋 Usage Guide

### Contract Analysis Methods

#### 1. Text Input
- Paste contract text directly into the interface
- Supports any plain text contract content
- Ideal for quick analysis and testing

#### 2. File Upload
- Upload PDF, TXT, or DOCX files
- Automatic text extraction and processing
- Supports multi-page documents

### Analysis Process
1. **Document Processing**: Upload or paste contract text
2. **Entity Extraction**: System identifies key parties, terms, and clauses
3. **Risk Assessment**: Analyzes potential legal and business risks
4. **Compliance Check**: Verifies against regulatory requirements
5. **Recommendations**: Generates actionable improvement suggestions

### Dashboard Features
- **Risk Dashboard**: Visual analytics for risk patterns
- **Analysis History**: Track and compare multiple analyses
- **Compliance Reports**: Detailed compliance scoring
- **System Metrics**: Performance and usage statistics

## 🔧 Configuration

### Environment Variables
```bash
# .env file
OLLAMA_MODEL=llama3.1:8b
CHROMA_DB_PATH=./chroma_db
KNOWLEDGE_BASE_PATH=./legal_knowledge_base
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

### Model Configuration
```python
# In main.py
OLLAMA_MODEL = "llama3.1:8b"  # Change to your preferred model
EMBEDDING_MODEL = "nomic-embed-text"
```

## 📊 Analysis Output

### Contract Analysis Results
```json
{
  "contract_type": "Service Agreement",
  "key_parties": ["ABC Consulting LLC", "XYZ Corporation"],
  "financial_terms": {
    "amounts": ["$10,000 per month"],
    "payment_terms": "Net 30"
  },
  "risks": [
    "Payment terms not clearly specified",
    "Termination clauses may be unclear"
  ],
  "recommendations": [
    "Clearly define payment terms including amounts, due dates, and late payment penalties",
    "Add liability limitation clauses to cap potential damages"
  ],
  "compliance_issues": [],
  "analysis_confidence": 0.85
}
```

### Risk Assessment Categories
- **High Risk**: Critical issues requiring immediate attention
- **Medium Risk**: Important concerns to address
- **Low Risk**: Minor issues for consideration
- **Compliance Issues**: Regulatory and legal compliance concerns

## 🛠️ Development

### Project Structure
```
legal-contract-analyzer/
├── main.py                 # Core analyzer logic
├── api.py                  # FastAPI backend
├── st_app.py              # Streamlit frontend
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── legal_knowledge_base/  # Legal documents and rules
├── chroma_db/            # Vector database storage
└── tests/                # Test files
```

### Key Classes
- **LegalContractAnalyzer**: Main analysis engine
- **ContractAnalyzerState**: LangGraph state management
- **ContractAnalysis**: Data structure for results
- **ContractAnalyzerUI**: Streamlit UI controller

## 📈 Performance Optimization

### Recommended Hardware
- **CPU**: 8+ cores for optimal performance
- **RAM**: 16GB+ (8GB minimum)
- **Storage**: SSD recommended for vector database
- **GPU**: Optional, but speeds up LLM inference

### Scaling Considerations
- Use GPU acceleration for large-scale processing
- Implement caching for repeated analyses
- Consider distributed processing for high-volume scenarios

## 🔒 Security & Privacy

### Data Handling
- **Local Processing**: All analysis happens locally
- **No External APIs**: Contracts never leave your environment
- **Secure Storage**: Vector database encryption available
- **Audit Trail**: Complete analysis history tracking

### Best Practices
- Regularly update the legal knowledge base
- Monitor system logs for security events
- Implement access controls for sensitive contracts
- Regular backup of analysis results

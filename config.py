import os
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama3.1:8b"
    embedding_model: str = "llama3.1:8b"
    temperature: float = 0.1
    max_tokens: int = 4096
    context_window: int = 8192

@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    persist_directory: str = "./chroma_db"
    collection_name: str = "legal_contracts"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_threshold: float = 0.7
    max_results: int = 5

@dataclass
class AnalysisConfig:
    """Configuration for contract analysis"""
    confidence_threshold: float = 0.6
    risk_categories: List[str] = None
    supported_file_types: List[str] = None
    max_file_size_mb: int = 50
    
    def __post_init__(self):
        if self.risk_categories is None:
            self.risk_categories = [
                "Financial Risk",
                "Legal Risk", 
                "Operational Risk",
                "Compliance Risk",
                "Reputational Risk"
            ]
        
        if self.supported_file_types is None:
            self.supported_file_types = ['.pdf', '.txt', '.docx', '.md']

@dataclass
class AppConfig:
    """Main application configuration"""
    app_name: str = "Legal Contract Analyzer"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    knowledge_base_path: str = "./legal_knowledge_base"
    output_directory: str = "./analysis_output"
    
    # Sub-configurations
    model: ModelConfig = None
    vectorstore: VectorStoreConfig = None
    analysis: AnalysisConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.vectorstore is None:
            self.vectorstore = VectorStoreConfig()
        if self.analysis is None:
            self.analysis = AnalysisConfig()

# Load configuration from environment variables
def load_config_from_env() -> AppConfig:
    """Load configuration from environment variables"""
    config = AppConfig()
    
    # Model configuration
    config.model.ollama_base_url = os.getenv("OLLAMA_BASE_URL", config.model.ollama_base_url)
    config.model.default_model = os.getenv("DEFAULT_MODEL", config.model.default_model)
    config.model.temperature = float(os.getenv("MODEL_TEMPERATURE", config.model.temperature))
    
    # Vector store configuration
    config.vectorstore.persist_directory = os.getenv("CHROMA_PERSIST_DIR", config.vectorstore.persist_directory)
    config.vectorstore.chunk_size = int(os.getenv("CHUNK_SIZE", config.vectorstore.chunk_size))
    
    # App configuration
    config.debug = os.getenv("DEBUG", "False").lower() == "true"
    config.log_level = os.getenv("LOG_LEVEL", config.log_level)
    config.knowledge_base_path = os.getenv("KNOWLEDGE_BASE_PATH", config.knowledge_base_path)
    
    return config

# Default configuration instance
CONFIG = load_config_from_env()

# Legal prompts and templates
LEGAL_PROMPTS = {
    "entity_extraction": """
    You are a legal contract analysis expert. Extract key information from the following contract:

    Contract Text: {contract_text}

    Extract and structure the following information:
    1. Contract Type (e.g., Service Agreement, Employment Contract, etc.)
    2. Parties Involved (names and roles)
    3. Effective Date and Term
    4. Key Financial Terms (amounts, payment schedules, penalties)
    5. Main Obligations of each party
    6. Termination Conditions
    7. Governing Law and Jurisdiction

    Format your response as structured data that can be easily parsed.
    """,
    
    "risk_assessment": """
    You are a legal risk assessment specialist. Analyze the following contract for potential risks:

    Contract Text: {contract_text}
    Contract Context: {context}

    Assess risks in these categories:
    1. Financial Risks (payment defaults, currency risks, penalty exposure)
    2. Legal Risks (compliance violations, jurisdiction issues, enforceability)
    3. Operational Risks (performance failures, delivery delays, quality issues)
    4. Strategic Risks (confidentiality breaches, competitive disadvantage)

    For each identified risk:
    - Describe the risk clearly
    - Assess severity (High/Medium/Low)
    - Suggest mitigation strategies

    Also identify any missing standard clauses that should be included.
    """,
    
    "compliance_check": """
    You are a legal compliance expert. Review this contract for compliance issues:

    Contract Text: {contract_text}
    Jurisdiction: {jurisdiction}
    Contract Type: {contract_type}

    Check compliance with:
    1. Contract formation requirements (offer, acceptance, consideration)
    2. Regulatory compliance (industry-specific regulations)
    3. Consumer protection laws (if applicable)
    4. Employment law compliance (if employment contract)
    5. Data protection and privacy requirements
    6. Anti-discrimination provisions
    7. Dispute resolution requirements

    Identify any compliance gaps and provide specific recommendations.
    """,
    
    "recommendation_generation": """
    You are a senior legal advisor. Based on the contract analysis, provide actionable recommendations:

    Contract Analysis Summary: {analysis_summary}
    Identified Risks: {risks}
    Compliance Issues: {compliance_issues}

    Provide specific, actionable recommendations in these areas:
    1. Critical Issues (must be addressed before signing)
    2. Contract Modifications (specific clause changes needed)
    3. Risk Mitigation Strategies
    4. Negotiation Points (areas for discussion with counterparty)
    5. Ongoing Monitoring Requirements

    Prioritize recommendations by importance and urgency.
    Format as numbered, actionable items.
    """,
    
    "clause_analysis": """
    You are a contract clause specialist. Analyze the following specific clauses:

    Contract Clauses: {clauses}
    Clause Type: {clause_type}

    Evaluate:
    1. Completeness (are all necessary elements present?)
    2. Clarity (is the language clear and unambiguous?)
    3. Fairness (is the clause balanced between parties?)
    4. Enforceability (is the clause legally enforceable?)
    5. Industry Standards (does it meet typical industry practices?)

    Provide specific suggestions for improvement.
    """
}

# Risk assessment criteria
RISK_CRITERIA = {
    "financial": {
        "high": [
            "unlimited liability",
            "no payment terms specified",
            "excessive penalties",
            "currency risk exposure",
            "no price adjustment mechanism"
        ],
        "medium": [
            "late payment penalties unclear",
            "payment schedule unrealistic",
            "no force majeure financial protection",
            "audit rights limited"
        ],
        "low": [
            "minor payment timing issues",
            "standard commercial terms",
            "reasonable penalty structure"
        ]
    },
    "legal": {
        "high": [
            "no governing law specified",
            "conflicting jurisdiction clauses",
            "regulatory compliance gaps",
            "intellectual property disputes",
            "indemnification imbalance"
        ],
        "medium": [
            "dispute resolution unclear",
            "confidentiality provisions weak",
            "termination process complex",
            "assignment restrictions unclear"
        ],
        "low": [
            "minor clause ambiguities",
            "standard legal provisions",
            "well-defined terms"
        ]
    },
    "operational": {
        "high": [
            "performance standards undefined",
            "delivery timelines unrealistic",
            "quality metrics missing",
            "no change management process",
            "inadequate service levels"
        ],
        "medium": [
            "reporting requirements unclear",
            "communication protocols vague",
            "escalation procedures missing",
            "resource allocation undefined"
        ],
        "low": [
            "minor process inefficiencies",
            "standard operational terms",
            "clear performance metrics"
        ]
    }
}

# Compliance frameworks
COMPLIANCE_FRAMEWORKS = {
    "general": [
        "Contract formation elements",
        "Capacity and authority",
        "Legal consideration",
        "Mutual assent",
        "Lawful purpose"
    ],
    "employment": [
        "Equal opportunity compliance",
        "Wage and hour laws",
        "Workplace safety requirements",
        "Non-discrimination provisions",
        "Privacy and confidentiality"
    ],
    "data_protection": [
        "GDPR compliance (if applicable)",
        "CCPA compliance (if applicable)",
        "Data processing agreements",
        "Privacy policy requirements",
        "Data breach notification"
    ],
    "financial_services": [
        "SOX compliance",
        "Anti-money laundering",
        "Know your customer (KYC)",
        "Consumer protection",
        "Regulatory reporting"
    ]
}

# Contract templates for knowledge base
CONTRACT_TEMPLATES = {
    "service_agreement": {
        "essential_clauses": [
            "Scope of Services",
            "Compensation and Payment Terms",
            "Term and Termination",
            "Intellectual Property Rights",
            "Confidentiality",
            "Limitation of Liability",
            "Governing Law"
        ],
        "recommended_clauses": [
            "Force Majeure",
            "Dispute Resolution",
            "Assignment and Subcontracting",
            "Compliance with Laws",
            "Insurance Requirements"
        ]
    },
    "employment_contract": {
        "essential_clauses": [
            "Job Description and Duties",
            "Compensation and Benefits",
            "Term of Employment",
            "Termination Conditions",
            "Confidentiality and Non-Disclosure",
            "Non-Competition (if applicable)",
            "Intellectual Property Assignment"
        ],
        "recommended_clauses": [
            "Performance Evaluation",
            "Training and Development",
            "Grievance Procedures",
            "Code of Conduct",
            "Data Protection"
        ]
    }
}

# Export key configurations
__all__ = [
    'CONFIG',
    'AppConfig',
    'ModelConfig', 
    'VectorStoreConfig',
    'AnalysisConfig',
    'LEGAL_PROMPTS',
    'RISK_CRITERIA',
    'COMPLIANCE_FRAMEWORKS',
    'CONTRACT_TEMPLATES'
]
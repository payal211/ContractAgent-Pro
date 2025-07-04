# Legal Contract Analyzer Agentic AI System
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import tool
from langchain.agents import Tool, AgentExecutor, create_react_agent

from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

import chromadb
from pathlib import Path
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContractAnalysis:
    """Data class for contract analysis results"""
    contract_type: str
    key_parties: List[str]
    key_terms: Dict[str, Any]
    risks: List[str]
    recommendations: List[str]
    compliance_issues: List[str]
    financial_terms: Dict[str, Any]
    termination_clauses: List[str]
    liability_clauses: List[str]
    confidentiality: Dict[str, Any]
    dispute_resolution: str
    governing_law: str
    analysis_confidence: float

class ContractAnalyzerState(TypedDict):
    """State for the LangGraph workflow"""
    messages: Annotated[list, add_messages]
    contract_text: str
    analysis_results: Optional[ContractAnalysis]
    current_step: str
    extracted_entities: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]

class LegalContractAnalyzer:
    """Main Legal Contract Analyzer using LangChain, RAG, and LangGraph"""
    
    def __init__(self, model_name: str = "llama3.1:8b", persist_directory: str = "./chroma_db"):
        self.model_name = model_name
        self.persist_directory = persist_directory
        
        # Initialize vector store first
        self.vectorstore = None
        self.retriever = None
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Legal knowledge base paths
        self.knowledge_base_path = "./legal_knowledge_base"
        
        # Initialize the system without Ollama for now
        self._setup_knowledge_base_without_llm()
        self._setup_langgraph()
    
    def _setup_knowledge_base_without_llm(self):
        """Setup the legal knowledge base without requiring Ollama"""
        try:
            # Create knowledge base directory if it doesn't exist
            Path(self.knowledge_base_path).mkdir(exist_ok=True)
            
            # Create sample legal knowledge
            self._create_sample_legal_knowledge()
            
            # Load legal documents
            documents = self._load_legal_documents()
            
            if documents:
                logger.info(f"Loaded {len(documents)} legal documents")
            else:
                logger.warning("No legal documents found.")
                
        except Exception as e:
            logger.error(f"Error setting up knowledge base: {e}")
    
    def _load_legal_documents(self) -> List[Document]:
        """Load legal documents from the knowledge base directory"""
        documents = []
        knowledge_path = Path(self.knowledge_base_path)
        
        for file_path in knowledge_path.glob("**/*"):
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                        docs = loader.load()
                        documents.extend(docs)
                    elif file_path.suffix.lower() in ['.txt', '.md']:
                        loader = TextLoader(str(file_path))
                        docs = loader.load()
                        documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _create_sample_legal_knowledge(self):
        """Create sample legal knowledge base"""
        sample_content = {
            "contract_types.txt": """
Contract Types and Classifications:

1. Service Agreements
- Professional services contracts
- Consulting agreements
- Maintenance contracts

2. Sales Contracts
- Purchase agreements
- Supply contracts
- Distribution agreements

3. Employment Contracts
- Employment agreements
- Non-disclosure agreements
- Non-compete agreements

4. Real Estate Contracts
- Lease agreements
- Purchase contracts
- Property management agreements

5. Technology Contracts
- Software licensing
- SaaS agreements
- Development contracts
""",
            "risk_factors.txt": """
Common Contract Risk Factors:

1. Financial Risks
- Payment terms unclear
- Penalty clauses excessive
- Currency fluctuation exposure
- Unlimited liability

2. Legal Risks
- Governing law conflicts
- Jurisdiction issues
- Compliance requirements
- Regulatory changes

3. Operational Risks
- Performance standards unclear
- Delivery timelines unrealistic
- Quality requirements undefined
- Force majeure provisions weak

4. Termination Risks
- Termination clauses unfavorable
- Notice periods inadequate
- Post-termination obligations
- Asset return requirements
""",
            "compliance_checklist.txt": """
Legal Compliance Checklist:

1. Contract Formation
- Offer and acceptance clear
- Consideration present
- Capacity to contract
- Legal purpose

2. Essential Terms
- Parties clearly identified
- Scope of work defined
- Payment terms specified
- Performance standards

3. Risk Mitigation
- Liability limitations
- Indemnification clauses
- Insurance requirements
- Force majeure provisions

4. Dispute Resolution
- Dispute resolution mechanism
- Governing law specified
- Jurisdiction clause
- Arbitration provisions
"""
        }
        
        knowledge_path = Path(self.knowledge_base_path)
        for filename, content in sample_content.items():
            with open(knowledge_path / filename, 'w') as f:
                f.write(content.strip())
    
    def _setup_langgraph(self):
        """Setup LangGraph workflow"""
        
        def extract_entities_node(state: ContractAnalyzerState):
            """Extract entities from contract"""
            contract_text = state["contract_text"]
            
            # Parse entities without LLM for now
            entities = {
                "parties": self._extract_parties(contract_text),
                "contract_type": self._extract_contract_type(contract_text),
                "key_dates": self._extract_dates(contract_text),
                "financial_terms": self._extract_financial_terms(contract_text)
            }
            
            return {
                **state,
                "extracted_entities": entities,
                "current_step": "risk_assessment"
            }
        
        def risk_assessment_node(state: ContractAnalyzerState):
            """Assess risks in the contract"""
            contract_text = state["contract_text"]
            entities = state["extracted_entities"]
            
            # Rule-based risk assessment
            risks = self._assess_risks_rule_based(contract_text, entities)
            
            risk_assessment = {
                "overall_risk_level": "Medium",
                "high_risks": risks.get("high_risks", []),
                "compliance_issues": risks.get("compliance_issues", []),
                "recommendations": []
            }
            
            return {
                **state,
                "risk_assessment": risk_assessment,
                "current_step": "generate_recommendations"
            }
        
        def recommendations_node(state: ContractAnalyzerState):
            """Generate recommendations"""
            contract_text = state["contract_text"]
            entities = state["extracted_entities"]
            risks = state["risk_assessment"]
            
            # Generate rule-based recommendations
            recommendations = self._generate_recommendations_rule_based(entities, risks)
            
            # Create final analysis
            analysis = ContractAnalysis(
                contract_type=entities.get("contract_type", "Unknown"),
                key_parties=entities.get("parties", []),
                key_terms=entities.get("financial_terms", {}),
                risks=risks.get("high_risks", []),
                recommendations=recommendations,
                compliance_issues=risks.get("compliance_issues", []),
                financial_terms=entities.get("financial_terms", {}),
                termination_clauses=self._extract_termination_clauses(contract_text),
                liability_clauses=self._extract_liability_clauses(contract_text),
                confidentiality=self._extract_confidentiality_terms(contract_text),
                dispute_resolution=self._extract_dispute_resolution(contract_text),
                governing_law=self._extract_governing_law(contract_text),
                analysis_confidence=0.8
            )
            
            return {
                **state,
                "analysis_results": analysis,
                "recommendations": recommendations,
                "current_step": "complete"
            }
        
        # Create the graph
        workflow = StateGraph(ContractAnalyzerState)
        
        # Add nodes
        workflow.add_node("extract_entities", extract_entities_node)
        workflow.add_node("risk_assessment", risk_assessment_node)
        workflow.add_node("generate_recommendations", recommendations_node)
        
        # Add edges
        workflow.set_entry_point("extract_entities")
        workflow.add_edge("extract_entities", "risk_assessment")
        workflow.add_edge("risk_assessment", "generate_recommendations")
        workflow.add_edge("generate_recommendations", END)
        
        # Compile the graph
        self.app = workflow.compile()
    
    def analyze_contract(self, contract_text: str) -> ContractAnalysis:
        """Main method to analyze a contract"""
        try:
            # Initialize state
            initial_state = ContractAnalyzerState(
                messages=[],
                contract_text=contract_text,
                analysis_results=None,
                current_step="start",
                extracted_entities={},
                risk_assessment={},
                recommendations=[]
            )
            
            # Run the workflow
            final_state = self.app.invoke(initial_state)
            
            return final_state["analysis_results"]
            
        except Exception as e:
            logger.error(f"Error analyzing contract: {e}")
            # Return default analysis on error
            return ContractAnalysis(
                contract_type="Unknown",
                key_parties=[],
                key_terms={},
                risks=[f"Analysis error: {str(e)}"],
                recommendations=["Please review contract manually"],
                compliance_issues=[],
                financial_terms={},
                termination_clauses=[],
                liability_clauses=[],
                confidentiality={},
                dispute_resolution="Unknown",
                governing_law="Unknown",
                analysis_confidence=0.0
            )
    
    def analyze_contract_file(self, file_path: str) -> ContractAnalysis:
        """Analyze a contract from a file"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
                contract_text = "\n".join([doc.page_content for doc in documents])
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    contract_text = f.read()
            
            return self.analyze_contract(contract_text)
            
        except Exception as e:
            logger.error(f"Error loading contract file: {e}")
            raise
    
    # Helper methods for parsing
    def _extract_parties(self, text: str) -> List[str]:
        """Extract parties from contract text"""
        patterns = [
            r"between\s+([^,\n]+)\s+and\s+([^,\n]+)",
            r"Party\s+1[:\s]+([^\n]+)",
            r"Party\s+2[:\s]+([^\n]+)",
            r'"([^"]+)"\s*\([^)]*\)',
            r"([A-Z][A-Za-z\s&.,]+(?:LLC|Corp|Corporation|Inc|Company))"
        ]
        
        parties = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    parties.extend([m for m in matches[0] if m.strip()])
                else:
                    parties.extend(matches)
        
        # Clean and deduplicate
        cleaned_parties = []
        for party in parties:
            party = party.strip().strip('"').strip("'")
            if party and len(party) > 2 and party not in cleaned_parties:
                cleaned_parties.append(party)
        
        return cleaned_parties[:5]  # Return max 5 parties
    
    def _extract_contract_type(self, text: str) -> str:
        """Extract contract type from text"""
        types = {
            "service agreement": ["service agreement", "consulting agreement", "professional services"],
            "employment contract": ["employment agreement", "employment contract"],
            "lease agreement": ["lease agreement", "rental agreement"],
            "purchase agreement": ["purchase agreement", "sales contract"],
            "license agreement": ["license agreement", "licensing agreement"],
            "partnership agreement": ["partnership agreement", "joint venture"],
            "distribution agreement": ["distribution agreement", "distributor agreement"],
            "non-disclosure agreement": ["non-disclosure", "nda", "confidentiality agreement"]
        }
        
        text_lower = text.lower()
        for contract_type, keywords in types.items():
            if any(keyword in text_lower for keyword in keywords):
                return contract_type.title()
        
        return "General Contract"
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract important dates from contract"""
        date_patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{1,2}\s+\w+\s+\d{4}\b",
            r"\b\w+\s+\d{1,2},\s+\d{4}\b",
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))[:10]  # Return unique dates, max 10
    
    def _extract_financial_terms(self, text: str) -> Dict[str, Any]:
        """Extract financial terms from contract"""
        financial_terms = {}
        
        # Extract amounts
        amount_patterns = [
            r"\$[\d,]+(?:\.\d{2})?",
            r"USD\s*[\d,]+(?:\.\d{2})?",
            r"[\d,]+\s*dollars?"
        ]
        
        amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts.extend(matches)
        
        if amounts:
            financial_terms["amounts"] = list(set(amounts))
        
        # Extract payment terms
        payment_patterns = [
            r"net\s+(\d+)",
            r"payment\s+terms?[:\s]+([^\n]+)",
            r"due\s+within\s+(\d+\s+days?)"
        ]
        
        for pattern in payment_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                financial_terms["payment_terms"] = matches[0]
                break
        
        return financial_terms
    
    def _extract_termination_clauses(self, text: str) -> List[str]:
        """Extract termination clauses"""
        termination_patterns = [
            r"(either party may terminate[^.]+\.)",
            r"(this agreement may be terminated[^.]+\.)",
            r"(termination[^.]+notice[^.]+\.)"
        ]
        
        clauses = []
        for pattern in termination_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            clauses.extend(matches)
        
        return clauses
    
    def _extract_liability_clauses(self, text: str) -> List[str]:
        """Extract liability clauses"""
        liability_patterns = [
            r"(liability[^.]+limited[^.]+\.)",
            r"(indemnify[^.]+\.)",
            r"(damages[^.]+limitation[^.]+\.)"
        ]
        
        clauses = []
        for pattern in liability_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            clauses.extend(matches)
        
        return clauses
    
    def _extract_confidentiality_terms(self, text: str) -> Dict[str, Any]:
        """Extract confidentiality terms"""
        confidentiality = {}
        
        if re.search(r"confidential", text, re.IGNORECASE):
            confidentiality["has_confidentiality"] = True
            
            # Look for specific terms
            if re.search(r"proprietary\s+information", text, re.IGNORECASE):
                confidentiality["covers_proprietary"] = True
            
            if re.search(r"non-disclosure", text, re.IGNORECASE):
                confidentiality["type"] = "Non-disclosure"
        else:
            confidentiality["has_confidentiality"] = False
        
        return confidentiality
    
    def _extract_dispute_resolution(self, text: str) -> str:
        """Extract dispute resolution mechanism"""
        if re.search(r"arbitration", text, re.IGNORECASE):
            return "Arbitration"
        elif re.search(r"mediation", text, re.IGNORECASE):
            return "Mediation"
        elif re.search(r"court", text, re.IGNORECASE):
            return "Court proceedings"
        else:
            return "Not specified"
    
    def _extract_governing_law(self, text: str) -> str:
        """Extract governing law"""
        law_patterns = [
            r"governed by the laws of\s+([^.\n]+)",
            r"laws of\s+([^.\n]+)\s+shall govern",
            r"jurisdiction of\s+([^.\n]+)"
        ]
        
        for pattern in law_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return "Not specified"
    
    def _assess_risks_rule_based(self, contract_text: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based risk assessment"""
        risks = {
            "high_risks": [],
            "medium_risks": [],
            "low_risks": [],
            "compliance_issues": []
        }
        
        text_lower = contract_text.lower()
        
        # Financial risks
        if not entities.get("financial_terms", {}).get("payment_terms"):
            risks["high_risks"].append("Payment terms not clearly specified")
        
        if "unlimited liability" in text_lower:
            risks["high_risks"].append("Unlimited liability exposure")
        
        # Termination risks
        if "terminate" not in text_lower:
            risks["medium_risks"].append("Termination clauses may be unclear")
        
        # Compliance risks
        if "governing law" not in text_lower:
            risks["compliance_issues"].append("Governing law not specified")
        
        if "dispute resolution" not in text_lower and "arbitration" not in text_lower:
            risks["compliance_issues"].append("Dispute resolution mechanism not specified")
        
        return risks
    
    def _generate_recommendations_rule_based(self, entities: Dict[str, Any], risks: Dict[str, Any]) -> List[str]:
        """Generate rule-based recommendations"""
        recommendations = []
        
        # Based on risks
        if "Payment terms not clearly specified" in risks.get("high_risks", []):
            recommendations.append("Clearly define payment terms including amounts, due dates, and late payment penalties")
        
        if "Unlimited liability exposure" in risks.get("high_risks", []):
            recommendations.append("Add liability limitation clauses to cap potential damages")
        
        if "Governing law not specified" in risks.get("compliance_issues", []):
            recommendations.append("Specify governing law and jurisdiction for dispute resolution")
        
        # Based on contract type
        contract_type = entities.get("contract_type", "").lower()
        if "service" in contract_type:
            recommendations.append("Include clear service level agreements (SLAs) and performance metrics")
            recommendations.append("Define scope of work in detail to avoid scope creep")
        
        if "employment" in contract_type:
            recommendations.append("Ensure compliance with local employment laws")
            recommendations.append("Include clear job description and reporting structure")
        
        # General recommendations
        recommendations.append("Review and update force majeure clauses")
        recommendations.append("Consider adding intellectual property protection clauses")
        
        return recommendations

def main():
    """Main function to demonstrate the contract analyzer"""
    # Initialize the analyzer (without Ollama for now)
    print("Initializing Legal Contract Analyzer...")
    analyzer = LegalContractAnalyzer()
    
    # Sample contract text for testing
    sample_contract = """
    SERVICE AGREEMENT
    
    This Service Agreement is entered into on January 15, 2024, between 
    ABC Consulting LLC ("Service Provider") and XYZ Corporation ("Client").
    
    1. SERVICES
    Service Provider agrees to provide consulting services as outlined in Exhibit A.
    
    2. COMPENSATION
    Client agrees to pay Service Provider $10,000 per month for services rendered.
    Payment terms are Net 30 from invoice date.
    
    3. TERM
    This agreement shall commence on February 1, 2024, and continue for 12 months.
    
    4. TERMINATION
    Either party may terminate this agreement with 30 days written notice.
    
    5. CONFIDENTIALITY
    Both parties agree to maintain confidentiality of all proprietary information.
    
    6. GOVERNING LAW
    This agreement shall be governed by the laws of California.
    """
    
    print("Legal Contract Analyzer - Starting Analysis...")
    print("=" * 50)
    
    # Analyze the sample contract
    analysis = analyzer.analyze_contract(sample_contract)
    
    # Display results
    print(f"Contract Type: {analysis.contract_type}")
    print(f"Key Parties: {', '.join(analysis.key_parties)}")
    print(f"Financial Terms: {analysis.financial_terms}")
    print(f"Governing Law: {analysis.governing_law}")
    print(f"Dispute Resolution: {analysis.dispute_resolution}")
    
    print("\nRisks Identified:")
    for risk in analysis.risks:
        print(f"  - {risk}")
    
    print("\nCompliance Issues:")
    for issue in analysis.compliance_issues:
        print(f"  - {issue}")
    
    print("\nRecommendations:")
    for rec in analysis.recommendations:
        print(f"  - {rec}")
    
    print(f"\nTermination Clauses:")
    for clause in analysis.termination_clauses:
        print(f"  - {clause}")
    
    print(f"\nConfidentiality: {analysis.confidentiality}")
    print(f"\nAnalysis Confidence: {analysis.analysis_confidence:.1%}")

if __name__ == "__main__":
    main()
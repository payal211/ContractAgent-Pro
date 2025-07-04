# FastAPI REST API for Legal Contract Analyzer

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import asyncio
import json
import uuid
import os
from datetime import datetime
from pathlib import Path
import logging

from main import LegalContractAnalyzer, ContractAnalysis
from config import CONFIG

# Configure logging
logging.basicConfig(level=getattr(logging, CONFIG.log_level))
logger = logging.getLogger(__name__)

# Global analyzer instance
analyzer = None

# Pydantic models
class ContractAnalysisRequest(BaseModel):
    contract_text: str
    analysis_type: Optional[str] = "full"
    priority: Optional[str] = "normal"

class ContractAnalysisResponse(BaseModel):
    analysis_id: str
    status: str
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
    timestamp: str

class AnalysisStatus(BaseModel):
    analysis_id: str
    status: str
    progress: int
    estimated_completion: Optional[str] = None
    error_message: Optional[str] = None

class RiskAssessmentResponse(BaseModel):
    analysis_id: str
    overall_risk_level: str
    risk_categories: Dict[str, List[str]]
    risk_scores: Dict[str, float]
    mitigation_strategies: List[str]

class ComplianceCheckResponse(BaseModel):
    analysis_id: str
    compliance_score: float
    compliance_issues: List[str]
    regulatory_requirements: List[str]
    recommendations: List[str]

# In-memory storage for analysis results (in production, use a database)
analysis_results = {}
analysis_status = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    global analyzer
    try:
        logger.info("Initializing Legal Contract Analyzer...")
        analyzer = LegalContractAnalyzer(
            model_name=CONFIG.model.default_model,
            persist_directory=CONFIG.vectorstore.persist_directory
        )
        logger.info("Analyzer initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Legal Contract Analyzer...")

# FastAPI app with lifespan
app = FastAPI(
    title=CONFIG.app_name,
    description="AI-powered legal contract analysis using LangChain, RAG, and LangGraph",
    version=CONFIG.version,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Legal Contract Analyzer API",
        "version": CONFIG.version,
        "status": "active",
        "endpoints": {
            "analyze": "/analyze",
            "analyze_file": "/analyze-file",
            "status": "/status/{analysis_id}",
            "results": "/results/{analysis_id}",
            "risk_assessment": "/risk-assessment/{analysis_id}",
            "compliance_check": "/compliance-check/{analysis_id}"
        }
    }

@app.post("/analyze")
async def analyze_contract(
    request: ContractAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Analyze contract text"""
    try:
        # Check if analyzer is initialized
        if analyzer is None:
            raise HTTPException(status_code=503, detail="Analyzer not initialized")
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Initialize status
        analysis_status[analysis_id] = AnalysisStatus(
            analysis_id=analysis_id,
            status="processing",
            progress=0
        )
        
        # Start background analysis
        background_tasks.add_task(
            perform_analysis,
            analysis_id,
            request.contract_text,
            request.analysis_type
        )
        
        return JSONResponse(
            content={
                "analysis_id": analysis_id,
                "status": "started",
                "message": "Analysis started. Check status endpoint for updates."
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-file")
async def analyze_contract_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Analyze contract from uploaded file"""
    try:
        # Check if analyzer is initialized
        if analyzer is None:
            raise HTTPException(status_code=503, detail="Analyzer not initialized")
        
        # Validate file type
        if not any(file.filename.lower().endswith(ext) for ext in CONFIG.analysis.supported_file_types):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {CONFIG.analysis.supported_file_types}"
            )
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{analysis_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize status
        analysis_status[analysis_id] = AnalysisStatus(
            analysis_id=analysis_id,
            status="processing",
            progress=0
        )
        
        # Start background analysis
        background_tasks.add_task(
            perform_file_analysis,
            analysis_id,
            str(file_path)
        )
        
        return JSONResponse(
            content={
                "analysis_id": analysis_id,
                "status": "started",
                "filename": file.filename,
                "message": "File analysis started. Check status endpoint for updates."
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting file analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{analysis_id}", response_model=AnalysisStatus)
async def get_analysis_status(analysis_id: str):
    """Get analysis status"""
    if analysis_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_status[analysis_id]

@app.get("/results/{analysis_id}", response_model=ContractAnalysisResponse)
async def get_analysis_results(analysis_id: str):
    """Get analysis results"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis results not found")
    
    if analysis_status[analysis_id].status != "completed":
        raise HTTPException(status_code=202, detail="Analysis not yet completed")
    
    return analysis_results[analysis_id]

@app.get("/risk-assessment/{analysis_id}", response_model=RiskAssessmentResponse)
async def get_risk_assessment(analysis_id: str):
    """Get detailed risk assessment"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[analysis_id]
    
    # Categorize risks
    risk_categories = {
        "high": [],
        "medium": [],
        "low": []
    }
    
    for risk in result["risks"]:
        # Simple categorization based on keywords
        if any(keyword in risk.lower() for keyword in ["critical", "severe", "major"]):
            risk_categories["high"].append(risk)
        elif any(keyword in risk.lower() for keyword in ["moderate", "medium"]):
            risk_categories["medium"].append(risk)
        else:
            risk_categories["low"].append(risk)
    
    return RiskAssessmentResponse(
        analysis_id=analysis_id,
        overall_risk_level="Medium",  # Would be calculated based on analysis
        risk_categories=risk_categories,
        risk_scores={
            "financial": 0.6,
            "legal": 0.4,
            "operational": 0.5,
            "compliance": 0.3
        },
        mitigation_strategies=result["recommendations"][:5]
    )

@app.get("/compliance-check/{analysis_id}", response_model=ComplianceCheckResponse)
async def get_compliance_check(analysis_id: str):
    """Get compliance check results"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[analysis_id]
    
    # Calculate compliance score (simplified)
    compliance_score = max(0.0, 1.0 - (len(result["compliance_issues"]) * 0.1))
    
    return ComplianceCheckResponse(
        analysis_id=analysis_id,
        compliance_score=compliance_score,
        compliance_issues=result["compliance_issues"],
        regulatory_requirements=[
            "Contract formation requirements",
            "Regulatory compliance standards",
            "Industry-specific regulations"
        ],
        recommendations=[rec for rec in result["recommendations"] if "compliance" in rec.lower()]
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": CONFIG.version,
        "analyzer_ready": analyzer is not None
    }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "total_analyses": len(analysis_results),
        "active_analyses": len([s for s in analysis_status.values() if s.status == "processing"]),
        "completed_analyses": len([s for s in analysis_status.values() if s.status == "completed"]),
        "failed_analyses": len([s for s in analysis_status.values() if s.status == "failed"]),
        "average_confidence": sum(r.get("analysis_confidence", 0) for r in analysis_results.values()) / max(len(analysis_results), 1)
    }

async def perform_analysis(analysis_id: str, contract_text: str, analysis_type: str):
    """Perform contract analysis in background"""
    try:
        # Update status
        analysis_status[analysis_id].status = "processing"
        analysis_status[analysis_id].progress = 25
        
        # Perform analysis
        analysis_result = analyzer.analyze_contract(contract_text)
        
        # Update status
        analysis_status[analysis_id].progress = 75
        
        # Convert to response format
        response = ContractAnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            contract_type=analysis_result.contract_type,
            key_parties=analysis_result.key_parties,
            key_terms=analysis_result.key_terms,
            risks=analysis_result.risks,
            recommendations=analysis_result.recommendations,
            compliance_issues=analysis_result.compliance_issues,
            financial_terms=analysis_result.financial_terms,
            termination_clauses=analysis_result.termination_clauses,
            liability_clauses=analysis_result.liability_clauses,
            confidentiality=analysis_result.confidentiality,
            dispute_resolution=analysis_result.dispute_resolution,
            governing_law=analysis_result.governing_law,
            analysis_confidence=analysis_result.analysis_confidence,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Store results
        analysis_results[analysis_id] = response.dict()
        
        # Update status
        analysis_status[analysis_id].status = "completed"
        analysis_status[analysis_id].progress = 100
        
        logger.info(f"Analysis {analysis_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {e}")
        analysis_status[analysis_id].status = "failed"
        analysis_status[analysis_id].error_message = str(e)

async def perform_file_analysis(analysis_id: str, file_path: str):
    """Perform file analysis in background"""
    try:
        # Update status
        analysis_status[analysis_id].status = "processing"
        analysis_status[analysis_id].progress = 25
        
        # Perform analysis
        analysis_result = analyzer.analyze_contract_file(file_path)
        
        # Update status
        analysis_status[analysis_id].progress = 75
        
        # Convert to response format
        response = ContractAnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            contract_type=analysis_result.contract_type,
            key_parties=analysis_result.key_parties,
            key_terms=analysis_result.key_terms,
            risks=analysis_result.risks,
            recommendations=analysis_result.recommendations,
            compliance_issues=analysis_result.compliance_issues,
            financial_terms=analysis_result.financial_terms,
            termination_clauses=analysis_result.termination_clauses,
            liability_clauses=analysis_result.liability_clauses,
            confidentiality=analysis_result.confidentiality,
            dispute_resolution=analysis_result.dispute_resolution,
            governing_law=analysis_result.governing_law,
            analysis_confidence=analysis_result.analysis_confidence,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Store results
        analysis_results[analysis_id] = response.dict()
        
        # Update status
        analysis_status[analysis_id].status = "completed"
        analysis_status[analysis_id].progress = 100
        
        # Clean up file
        try:
            os.remove(file_path)
        except:
            pass
        
        logger.info(f"File analysis {analysis_id} completed successfully")
        
    except Exception as e:
        logger.error(f"File analysis {analysis_id} failed: {e}")
        analysis_status[analysis_id].status = "failed"
        analysis_status[analysis_id].error_message = str(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
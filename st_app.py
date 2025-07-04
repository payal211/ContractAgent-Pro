# Streamlit Web UI for Legal Contract Analyzer

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import io
from datetime import datetime
import base64

# Page configuration
st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .confidence-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .recommendation-item {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

class ContractAnalyzerUI:
    def __init__(self):
        self.api_base = API_BASE_URL
        
    def check_api_health(self):
        """Check if API is available"""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def analyze_contract_text(self, contract_text: str) -> str:
        """Submit contract text for analysis"""
        try:
            response = requests.post(
                f"{self.api_base}/analyze",
                json={"contract_text": contract_text},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["analysis_id"]
            else:
                st.error(f"API Error: {response.text}")
                return None
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
            return None
    
    def analyze_contract_file(self, file_content, filename: str) -> str:
        """Submit contract file for analysis"""
        try:
            files = {"file": (filename, file_content, "application/octet-stream")}
            response = requests.post(
                f"{self.api_base}/analyze-file",
                files=files,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["analysis_id"]
            else:
                st.error(f"API Error: {response.text}")
                return None
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
            return None
    
    def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis status"""
        try:
            response = requests.get(f"{self.api_base}/status/{analysis_id}")
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_analysis_results(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis results"""
        try:
            response = requests.get(f"{self.api_base}/results/{analysis_id}")
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_risk_assessment(self, analysis_id: str) -> Dict[str, Any]:
        """Get risk assessment"""
        try:
            response = requests.get(f"{self.api_base}/risk-assessment/{analysis_id}")
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_compliance_check(self, analysis_id: str) -> Dict[str, Any]:
        """Get compliance check"""
        try:
            response = requests.get(f"{self.api_base}/compliance-check/{analysis_id}")
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            response = requests.get(f"{self.api_base}/metrics")
            return response.json() if response.status_code == 200 else None
        except:
            return None

def main():
    # Initialize the UI
    analyzer_ui = ContractAnalyzerUI()
    
    # Header
    st.markdown('<h1 class="main-header">⚖️ Legal Contract Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Contract Analysis using LangChain, RAG, and LangGraph**")
    
    # Check API health
    if not analyzer_ui.check_api_health():
        st.error("🚨 Cannot connect to the API server. Please ensure the API is running on http://localhost:8000")
        st.info("Run the API with: `python api.py` or `uvicorn api:app --reload`")
        return
    
    st.success("✅ Connected to API server")
    
    # Initialize session state
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Contract Analysis",
        "Analysis History", 
        "Risk Dashboard",
        "Compliance Check",
        "System Metrics",
        "About"
    ])
    
    if page == "Contract Analysis":
        contract_analysis_page(analyzer_ui)
    elif page == "Analysis History":
        analysis_history_page(analyzer_ui)
    elif page == "Risk Dashboard":
        risk_dashboard_page(analyzer_ui)
    elif page == "Compliance Check":
        compliance_check_page(analyzer_ui)
    elif page == "System Metrics":
        system_metrics_page(analyzer_ui)
    else:
        about_page()

def contract_analysis_page(analyzer_ui: ContractAnalyzerUI):
    """Main contract analysis page"""
    st.header("📄 Contract Analysis")
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Text Input", "File Upload"])
    
    analysis_id = None
    
    if input_method == "Text Input":
        # Text input
        st.subheader("Contract Text")
        contract_text = st.text_area(
            "Paste your contract text here:",
            height=300,
            placeholder="Enter the contract text you want to analyze..."
        )
        
        if st.button("Analyze Contract", type="primary"):
            if contract_text.strip():
                with st.spinner("Submitting contract for analysis..."):
                    analysis_id = analyzer_ui.analyze_contract_text(contract_text)
                    if analysis_id:
                        st.session_state.current_analysis_id = analysis_id
                        st.success(f"Analysis started! Analysis ID: {analysis_id}")
                        # Add to history
                        st.session_state.analysis_history.append({
                            "id": analysis_id,
                            "timestamp": datetime.now().isoformat(),
                            "type": "Text Input",
                            "status": "processing"
                        })
            else:
                st.warning("Please enter contract text before analyzing.")
    
    else:
        # File upload
        st.subheader("Upload Contract File")
        uploaded_file = st.file_uploader(
            "Choose a contract file",
            type=['pdf', 'txt', 'docx'],
            help="Supported formats: PDF, TXT, DOCX"
        )
        
        if uploaded_file is not None:
            st.info(f"File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            if st.button("Analyze File", type="primary"):
                with st.spinner("Uploading and analyzing file..."):
                    analysis_id = analyzer_ui.analyze_contract_file(
                        uploaded_file.getvalue(),
                        uploaded_file.name
                    )
                    if analysis_id:
                        st.session_state.current_analysis_id = analysis_id
                        st.success(f"File analysis started! Analysis ID: {analysis_id}")
                        # Add to history
                        st.session_state.analysis_history.append({
                            "id": analysis_id,
                            "timestamp": datetime.now().isoformat(),
                            "type": f"File Upload ({uploaded_file.name})",
                            "status": "processing"
                        })
    
    # Show analysis results if available
    if hasattr(st.session_state, 'current_analysis_id'):
        show_analysis_progress(analyzer_ui, st.session_state.current_analysis_id)

def show_analysis_progress(analyzer_ui: ContractAnalyzerUI, analysis_id: str):
    """Show analysis progress and results"""
    st.subheader("Analysis Progress")
    
    # Create progress container
    progress_container = st.container()
    
    # Poll for status
    with progress_container:
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        max_polls = 60  # Maximum 60 polls (5 minutes with 5-second intervals)
        poll_count = 0
        
        while poll_count < max_polls:
            status = analyzer_ui.get_analysis_status(analysis_id)
            
            if status:
                progress_bar.progress(status.get("progress", 0) / 100)
                status_placeholder.info(f"Status: {status['status']} - Progress: {status.get('progress', 0)}%")
                
                if status["status"] == "completed":
                    st.success("✅ Analysis completed!")
                    show_analysis_results(analyzer_ui, analysis_id)
                    break
                elif status["status"] == "failed":
                    st.error(f"❌ Analysis failed: {status.get('error_message', 'Unknown error')}")
                    break
            
            time.sleep(5)  # Wait 5 seconds before next poll
            poll_count += 1
        
        if poll_count >= max_polls:
            st.warning("Analysis is taking longer than expected. Please check back later.")

def show_analysis_results(analyzer_ui: ContractAnalyzerUI, analysis_id: str):
    """Display analysis results"""
    results = analyzer_ui.get_analysis_results(analysis_id)
    
    if not results:
        st.error("Failed to retrieve analysis results")
        return
    
    st.header("📊 Analysis Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Contract Type", results.get("contract_type", "Unknown"))
    
    with col2:
        st.metric("Number of Parties", len(results.get("key_parties", [])))
    
    with col3:
        st.metric("Risks Identified", len(results.get("risks", [])))
    
    with col4:
        confidence = results.get("analysis_confidence", 0)
        st.metric("Confidence Score", f"{confidence:.1%}")
    
    # Detailed sections
    st.subheader("🏢 Key Parties")
    if results.get("key_parties"):
        for i, party in enumerate(results["key_parties"], 1):
            st.write(f"{i}. {party}")
    else:
        st.info("No parties identified")
    
    # Financial Terms
    st.subheader("💰 Financial Terms")
    financial_terms = results.get("financial_terms", {})
    if financial_terms:
        st.json(financial_terms)
    else:
        st.info("No financial terms identified")
    
    # Risks
    st.subheader("⚠️ Identified Risks")
    risks = results.get("risks", [])
    if risks:
        for risk in risks:
            risk_level = "high" if any(keyword in risk.lower() for keyword in ["critical", "severe", "major"]) else "medium"
            st.markdown(f'<div class="risk-{risk_level}">{risk}</div>', unsafe_allow_html=True)
    else:
        st.success("No significant risks identified")
    
    # Recommendations
    st.subheader("📋 Recommendations")
    recommendations = results.get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f'<div class="recommendation-item"><strong>{i}.</strong> {rec}</div>', unsafe_allow_html=True)
    else:
        st.info("No specific recommendations")
    
    # Compliance Issues
    st.subheader("📜 Compliance Issues")
    compliance_issues = results.get("compliance_issues", [])
    if compliance_issues:
        for issue in compliance_issues:
            st.warning(f"⚠️ {issue}")
    else:
        st.success("✅ No compliance issues identified")
    
    # Contract Details
    with st.expander("🔍 Additional Contract Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Governing Law:**", results.get("governing_law", "Not specified"))
            st.write("**Dispute Resolution:**", results.get("dispute_resolution", "Not specified"))
        
        with col2:
            st.write("**Analysis ID:**", results.get("analysis_id", "Unknown"))
            st.write("**Timestamp:**", results.get("timestamp", "Unknown"))
    
    # Export functionality
    st.subheader("📥 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download JSON Report"):
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"contract_analysis_{analysis_id}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Generate PDF Report"):
            st.info("PDF generation feature coming soon!")

def analysis_history_page(analyzer_ui: ContractAnalyzerUI):
    """Analysis history page"""
    st.header("📚 Analysis History")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history available. Start by analyzing a contract!")
        return
    
    # Display history as table
    history_df = pd.DataFrame(st.session_state.analysis_history)
    st.dataframe(history_df, use_container_width=True)
    
    # Select analysis to view
    st.subheader("View Previous Analysis")
    selected_id = st.selectbox(
        "Select an analysis to view:",
        options=[item["id"] for item in st.session_state.analysis_history],
        format_func=lambda x: f"{x[:8]}... - {next(item['type'] for item in st.session_state.analysis_history if item['id'] == x)}"
    )
    
    if selected_id and st.button("Load Analysis"):
        show_analysis_results(analyzer_ui, selected_id)

def risk_dashboard_page(analyzer_ui: ContractAnalyzerUI):
    """Risk dashboard page"""
    st.header("📊 Risk Dashboard")
    
    if not st.session_state.analysis_history:
        st.info("No analysis data available for dashboard. Start by analyzing some contracts!")
        return
    
    # Get completed analyses
    completed_analyses = [
        item for item in st.session_state.analysis_history 
        if item.get("status") == "completed"
    ]
    
    if not completed_analyses:
        st.info("No completed analyses available for dashboard.")
        return
    
    # Risk overview charts
    st.subheader("Risk Overview")
    
    # Create sample risk data (in real implementation, this would come from actual analysis results)
    risk_categories = ["Financial", "Legal", "Operational", "Compliance"]
    risk_counts = [12, 8, 15, 5]  # Sample data
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution pie chart
        fig_pie = px.pie(
            values=risk_counts,
            names=risk_categories,
            title="Risk Distribution by Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Risk severity bar chart
        severity_data = {
            "Risk Level": ["High", "Medium", "Low"],
            "Count": [8, 15, 17]  # Sample data
        }
        fig_bar = px.bar(
            severity_data,
            x="Risk Level",
            y="Count",
            title="Risk Distribution by Severity",
            color="Risk Level",
            color_discrete_map={"High": "#ff4444", "Medium": "#ffaa00", "Low": "#44ff44"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Risk trends over time
    st.subheader("Risk Trends")
    
    # Sample trend data
    dates = pd.date_range(start="2024-01-01", periods=10, freq="W")
    risk_trend_data = {
        "Date": dates,
        "High Risk": [2, 3, 1, 4, 2, 3, 5, 2, 1, 3],
        "Medium Risk": [5, 4, 6, 3, 5, 4, 6, 7, 5, 4],
        "Low Risk": [8, 7, 9, 6, 8, 9, 7, 8, 9, 8]
    }
    
    trend_df = pd.DataFrame(risk_trend_data)
    fig_trend = px.line(
        trend_df,
        x="Date",
        y=["High Risk", "Medium Risk", "Low Risk"],
        title="Risk Trends Over Time",
        markers=True
    )
    st.plotly_chart(fig_trend, use_container_width=True)

def compliance_check_page(analyzer_ui: ContractAnalyzerUI):
    """Compliance check page"""
    st.header("📜 Compliance Check")
    
    if not st.session_state.analysis_history:
        st.info("No analysis data available. Start by analyzing some contracts!")
        return
    
    # Select analysis for compliance check
    analysis_ids = [item["id"] for item in st.session_state.analysis_history]
    selected_analysis = st.selectbox(
        "Select analysis for compliance check:",
        options=analysis_ids,
        format_func=lambda x: f"{x[:8]}..."
    )
    
    if selected_analysis and st.button("Run Compliance Check"):
        compliance_data = analyzer_ui.get_compliance_check(selected_analysis)
        
        if compliance_data:
            st.subheader("Compliance Results")
            
            # Compliance score
            score = compliance_data.get("compliance_score", 0)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Compliance Score", f"{score:.1%}")
            
            with col2:
                issues_count = len(compliance_data.get("compliance_issues", []))
                st.metric("Issues Found", issues_count)
            
            with col3:
                requirements_count = len(compliance_data.get("regulatory_requirements", []))
                st.metric("Requirements", requirements_count)
            
            # Compliance gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Compliance Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Issues and recommendations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Compliance Issues")
                issues = compliance_data.get("compliance_issues", [])
                if issues:
                    for issue in issues:
                        st.warning(f"⚠️ {issue}")
                else:
                    st.success("✅ No compliance issues found")
            
            with col2:
                st.subheader("Recommendations")
                recommendations = compliance_data.get("recommendations", [])
                if recommendations:
                    for rec in recommendations:
                        st.info(f"💡 {rec}")
                else:
                    st.info("No specific recommendations")

def system_metrics_page(analyzer_ui: ContractAnalyzerUI):
    """System metrics page"""
    st.header("📈 System Metrics")
    
    metrics = analyzer_ui.get_metrics()
    
    if not metrics:
        st.error("Unable to retrieve system metrics")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", metrics.get("total_analyses", 0))
    
    with col2:
        st.metric("Active Analyses", metrics.get("active_analyses", 0))
    
    with col3:
        st.metric("Completed", metrics.get("completed_analyses", 0))
    
    with col4:
        st.metric("Failed", metrics.get("failed_analyses", 0))
    
    # Success rate
    total = metrics.get("total_analyses", 1)
    completed = metrics.get("completed_analyses", 0)
    success_rate = (completed / total) * 100 if total > 0 else 0
    
    st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Average confidence
    avg_confidence = metrics.get("average_confidence", 0)
    st.metric("Average Confidence", f"{avg_confidence:.1%}")
    
    # System status visualization
    status_data = {
        "Status": ["Completed", "Active", "Failed"],
        "Count": [
            metrics.get("completed_analyses", 0),
            metrics.get("active_analyses", 0),
            metrics.get("failed_analyses", 0)
        ]
    }
    
    fig_status = px.bar(
        status_data,
        x="Status",
        y="Count",
        title="Analysis Status Distribution",
        color="Status",
        color_discrete_map={
            "Completed": "#44ff44",
            "Active": "#ffaa00", 
            "Failed": "#ff4444"
        }
    )
    st.plotly_chart(fig_status, use_container_width=True)

def about_page():
    """About page"""
    st.header("ℹ️ About Legal Contract Analyzer")
    
    st.markdown("""
    ## Overview
    The Legal Contract Analyzer is an AI-powered system that uses advanced natural language processing 
    to analyze legal contracts and identify potential risks, compliance issues, and provide actionable recommendations.
    
    ## Features
    - **Contract Analysis**: Automated analysis of contract text and files
    - **Risk Assessment**: Identification and categorization of potential risks
    - **Compliance Checking**: Verification against legal and regulatory requirements
    - **Recommendations**: Actionable suggestions for contract improvement
    - **Multi-format Support**: PDF, TXT, and DOCX file formats
    - **Real-time Processing**: Live status updates during analysis
    
    ## Technology Stack
    - **LangChain**: For building the AI pipeline
    - **LangGraph**: For workflow orchestration
    - **RAG (Retrieval Augmented Generation)**: For knowledge-enhanced analysis
    - **Ollama**: For local LLM deployment
    - **FastAPI**: For the backend API
    - **Streamlit**: For the web interface
    - **ChromaDB**: For vector storage and retrieval
    
    ## How It Works
    1. **Document Processing**: Upload or paste contract text
    2. **Entity Extraction**: Identify key parties, terms, and clauses
    3. **Risk Assessment**: Analyze potential legal and business risks
    4. **Compliance Check**: Verify against regulatory requirements
    5. **Recommendations**: Generate actionable improvement suggestions
    
    ## Disclaimer
    This tool is designed to assist legal professionals and should not replace 
    professional legal advice. Always consult with qualified legal counsel for 
    important contract decisions.
    
    ## Version
    Current Version: 1.0.0
    """)
    
    st.subheader("📞 Support")
    st.info("For technical support or questions, please contact the development team.")

if __name__ == "__main__":
    main()
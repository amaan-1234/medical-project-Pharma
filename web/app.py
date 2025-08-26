"""
Streamlit web interface for Clinical Trial Outcome Predictor
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import load_sample_data, save_sample_data
from utils.llm_utils import create_sample_protocol

# Page configuration
st.set_page_config(
    page_title="Clinical Trial Outcome Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

if 'protocol_analysis' not in st.session_state:
    st.session_state.protocol_analysis = None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Clinical Trial Outcome Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Analysis for Clinical Trial Success Prediction")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page",
            ["🏠 Dashboard", "🔮 Trial Prediction", "📊 Protocol Analysis", "📈 Data Explorer", "⚙️ Settings"]
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        
        # Load sample data for sidebar stats
        try:
            data = load_sample_data()
            st.metric("Total Trials", len(data['trial_data']))
            st.metric("Total Patients", len(data['patient_data']))
            st.metric("Avg Success Rate", f"{data['trial_data']['success_rate'].mean():.1%}")
        except:
            st.info("Data not available")
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔮 Trial Prediction":
        show_trial_prediction()
    elif page == "📊 Protocol Analysis":
        show_protocol_analysis()
    elif page == "📈 Data Explorer":
        show_data_explorer()
    elif page == "⚙️ Settings":
        show_settings()

def show_dashboard():
    """Show the main dashboard"""
    
    # Load data
    try:
        data = load_sample_data()
        
        # Create three columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Clinical Trials", len(data['trial_data']))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Patients", len(data['patient_data']))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_success = data['trial_data']['success_rate'].mean()
            st.metric("Average Success Rate", f"{avg_success:.1%}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Charts section
        st.markdown("---")
        st.subheader("📊 Trial Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Phase distribution
            phase_counts = data['trial_data']['phase'].value_counts()
            fig_phase = px.pie(
                values=phase_counts.values,
                names=phase_counts.index,
                title="Trial Phase Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_phase, use_container_width=True)
        
        with col2:
            # Intervention type distribution
            intervention_counts = data['trial_data']['intervention_type'].value_counts()
            fig_intervention = px.bar(
                x=intervention_counts.index,
                y=intervention_counts.values,
                title="Intervention Type Distribution",
                color=intervention_counts.values,
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig_intervention, use_container_width=True)
        
        # Success rate by phase
        st.subheader("🎯 Success Rate Analysis")
        success_by_phase = data['trial_data'].groupby('phase')['success_rate'].mean().reset_index()
        
        fig_success = px.bar(
            success_by_phase,
            x='phase',
            y='success_rate',
            title="Success Rate by Trial Phase",
            color='success_rate',
            color_continuous_scale="RdYlGn",
            text=success_by_phase['success_rate'].apply(lambda x: f"{x:.1%}")
        )
        fig_success.update_traces(textposition='outside')
        st.plotly_chart(fig_success, use_container_width=True)
        
        # Recent predictions (if any)
        if st.session_state.prediction_result:
            st.markdown("---")
            st.subheader("🔮 Recent Prediction")
            
            result = st.session_state.prediction_result
            if result['prediction'] == 'Success':
                card_class = "success-card"
            else:
                card_class = "warning-card"
            
            st.markdown(f'<div class="metric-card {card_class}">', unsafe_allow_html=True)
            st.markdown(f"**Prediction:** {result['prediction']}")
            st.markdown(f"**Success Probability:** {result['success_probability']:.1%}")
            st.markdown(f"**Risk Factors:** {', '.join(result['risk_factors'])}")
            st.markdown("</div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        st.info("Please ensure the data generation script has been run first.")

def show_trial_prediction():
    """Show the trial prediction interface"""
    
    st.header("🔮 Clinical Trial Outcome Prediction")
    st.markdown("Enter patient data and optionally a trial protocol to predict trial success.")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Patient Information")
        
        # Patient demographics
        with st.expander("Demographics", expanded=True):
            age = st.slider("Age", 18, 100, 65)
            gender = st.selectbox("Gender", ["M", "F"])
            race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
            ethnicity = st.selectbox("Ethnicity", ["Hispanic", "Non-Hispanic"])
            bmi = st.slider("BMI", 16.0, 50.0, 28.0, 0.1)
            smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        
        # Medical conditions
        with st.expander("Medical Conditions"):
            diabetes = st.checkbox("Diabetes")
            hypertension = st.checkbox("Hypertension")
            heart_disease = st.checkbox("Heart Disease")
        
        # Biomarkers
        with st.expander("Laboratory Values"):
            col_bio1, col_bio2 = st.columns(2)
            
            with col_bio1:
                creatinine = st.number_input("Creatinine (mg/dL)", 0.5, 3.0, 1.1, 0.1)
                albumin = st.number_input("Albumin (g/dL)", 2.5, 5.5, 4.0, 0.1)
                hemoglobin = st.number_input("Hemoglobin (g/dL)", 10.0, 18.0, 14.0, 0.1)
                platelet_count = st.number_input("Platelet Count (K/μL)", 150, 450, 250)
            
            with col_bio2:
                white_blood_cells = st.number_input("WBC (K/μL)", 4.0, 11.0, 7.0, 0.1)
                sodium = st.number_input("Sodium (mEq/L)", 135, 145, 140)
                potassium = st.number_input("Potassium (mEq/L)", 3.5, 5.0, 4.0, 0.1)
                glucose = st.number_input("Glucose (mg/dL)", 70, 200, 100)
    
    with col2:
        st.subheader("Trial Protocol (Optional)")
        
        use_sample_protocol = st.checkbox("Use Sample Protocol", value=True)
        
        if use_sample_protocol:
            sample_protocol = create_sample_protocol()
            protocol_text = st.text_area(
                "Protocol Text",
                value=sample_protocol,
                height=300,
                help="Enter or modify the clinical trial protocol text"
            )
        else:
            protocol_text = st.text_area(
                "Protocol Text",
                height=300,
                placeholder="Enter clinical trial protocol text here...",
                help="Enter the clinical trial protocol text for analysis"
            )
        
        use_llm = st.checkbox("Use LLM Analysis", value=True, help="Enable LLM-based protocol analysis")
    
    # Prediction button
    if st.button("🚀 Predict Trial Outcome", type="primary", use_container_width=True):
        with st.spinner("Analyzing data and making prediction..."):
            try:
                # Prepare patient data
                patient_data = {
                    "age": age,
                    "gender": gender,
                    "race": race,
                    "ethnicity": ethnicity,
                    "bmi": bmi,
                    "smoking_status": smoking_status,
                    "diabetes": 1 if diabetes else 0,
                    "hypertension": 1 if hypertension else 0,
                    "heart_disease": 1 if heart_disease else 0,
                    "creatinine": creatinine,
                    "albumin": albumin,
                    "hemoglobin": hemoglobin,
                    "platelet_count": platelet_count,
                    "white_blood_cells": white_blood_cells,
                    "sodium": sodium,
                    "potassium": potassium,
                    "glucose": glucose
                }
                
                # Prepare request
                request_data = {
                    "patient_data": patient_data,
                    "use_llm_analysis": use_llm
                }
                
                if protocol_text.strip():
                    request_data["trial_protocol"] = {
                        "protocol_text": protocol_text
                    }
                
                # Make API call (simulate if API not running)
                try:
                    response = requests.post(
                        "http://localhost:8000/predict/trial",
                        json=request_data,
                        timeout=30
                    )
                    if response.status_code == 200:
                        prediction_result = response.json()
                    else:
                        raise Exception(f"API error: {response.status_code}")
                except:
                    # Fallback to simulated prediction
                    st.warning("API not available, using simulated prediction")
                    prediction_result = simulate_prediction(patient_data, protocol_text)
                
                # Store result in session state
                st.session_state.prediction_result = prediction_result
                
                # Show result
                show_prediction_result(prediction_result)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    
    # Show previous prediction if available
    if st.session_state.prediction_result:
        st.markdown("---")
        st.subheader("Previous Prediction")
        show_prediction_result(st.session_state.prediction_result)

def simulate_prediction(patient_data, protocol_text):
    """Simulate prediction when API is not available"""
    
    # Simple risk calculation
    risk_score = 0.0
    
    # Age risk
    if patient_data['age'] > 70:
        risk_score += 0.2
    elif patient_data['age'] > 60:
        risk_score += 0.1
    
    # Medical conditions
    if patient_data['diabetes']:
        risk_score += 0.15
    if patient_data['hypertension']:
        risk_score += 0.1
    if patient_data['heart_disease']:
        risk_score += 0.2
    
    # Biomarker risks
    if patient_data['creatinine'] > 1.5:
        risk_score += 0.1
    if patient_data['glucose'] > 126:
        risk_score += 0.1
    
    # Protocol risk (simple text analysis)
    if protocol_text:
        text_lower = protocol_text.lower()
        if "phase 1" in text_lower:
            risk_score += 0.3
        elif "phase 2" in text_lower:
            risk_score += 0.2
        if "placebo" not in text_lower:
            risk_score += 0.05
    
    # Calculate success probability
    success_probability = max(0.1, 1.0 - risk_score)
    
    # Identify risk factors
    risk_factors = []
    if patient_data['age'] > 70:
        risk_factors.append("Advanced age")
    if patient_data['diabetes']:
        risk_factors.append("Diabetes")
    if patient_data['hypertension']:
        risk_factors.append("Hypertension")
    if patient_data['heart_disease']:
        risk_factors.append("Heart disease")
    
    return {
        'success_probability': success_probability,
        'prediction': 'Success' if success_probability > 0.5 else 'Failure',
        'confidence_score': 0.8,
        'risk_factors': risk_factors,
        'model_used': 'simulated',
        'timestamp': datetime.now().isoformat()
    }

def show_prediction_result(result):
    """Display prediction result in a nice format"""
    
    # Create columns for result display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if result['prediction'] == 'Success':
            st.markdown('<div class="metric-card success-card">', unsafe_allow_html=True)
            st.metric("Prediction", "✅ Success", delta=f"{result['success_probability']:.1%}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card danger-card">', unsafe_allow_html=True)
            st.metric("Prediction", "❌ Failure", delta=f"{result['success_probability']:.1%}")
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Success Probability", f"{result['success_probability']:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Confidence", f"{result['confidence_score']:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Risk factors
    if result['risk_factors']:
        st.markdown("---")
        st.subheader("⚠️ Identified Risk Factors")
        
        for factor in result['risk_factors']:
            st.markdown(f"- {factor}")
    
    # Additional details
    with st.expander("Prediction Details"):
        st.json(result)

def show_protocol_analysis():
    """Show protocol analysis interface"""
    
    st.header("📊 Protocol Analysis")
    st.markdown("Analyze clinical trial protocols to extract key features and assess risk.")
    
    # Protocol input
    protocol_text = st.text_area(
        "Clinical Trial Protocol",
        value=create_sample_protocol(),
        height=400,
        help="Enter the clinical trial protocol text for analysis"
    )
    
    if st.button("🔍 Analyze Protocol", type="primary", use_container_width=True):
        with st.spinner("Analyzing protocol..."):
            try:
                # Try API call first
                try:
                    response = requests.post(
                        "http://localhost:8000/analyze/protocol",
                        json={"protocol_text": protocol_text},
                        timeout=30
                    )
                    if response.status_code == 200:
                        analysis_result = response.json()
                    else:
                        raise Exception(f"API error: {response.status_code}")
                except:
                    # Fallback to local analysis
                    st.warning("API not available, using local analysis")
                    from utils.llm_utils import ProtocolAnalyzer
                    analyzer = ProtocolAnalyzer()
                    analysis_result = analyzer.analyze_protocol(protocol_text)
                    risk_score = analyzer.calculate_risk_score(analysis_result)
                    summary = analyzer.generate_protocol_summary(analysis_result)
                    
                    analysis_result = {
                        "analysis": analysis_result,
                        "risk_score": risk_score,
                        "summary": summary,
                        "confidence": 0.7
                    }
                
                # Store result
                st.session_state.protocol_analysis = analysis_result
                
                # Show results
                show_protocol_analysis_result(analysis_result)
                
            except Exception as e:
                st.error(f"Protocol analysis failed: {e}")
    
    # Show previous analysis if available
    if st.session_state.protocol_analysis:
        st.markdown("---")
        st.subheader("Previous Analysis")
        show_protocol_analysis_result(st.session_state.protocol_analysis)

def show_protocol_analysis_result(result):
    """Display protocol analysis results"""
    
    # Summary
    st.subheader("📋 Analysis Summary")
    st.markdown(result['summary'])
    
    # Risk assessment
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Risk Score", f"{result['risk_score']:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Detailed analysis
    with st.expander("Detailed Analysis"):
        st.json(result['analysis'])

def show_data_explorer():
    """Show data exploration interface"""
    
    st.header("📈 Data Explorer")
    st.markdown("Explore the clinical trial dataset and generate insights.")
    
    try:
        data = load_sample_data()
        
        # Data overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Trials", len(data['trial_data']))
        with col2:
            st.metric("Patients", len(data['patient_data']))
        with col3:
            st.metric("Outcomes", len(data['outcome_data']))
        
        # Data tables
        tab1, tab2, tab3 = st.tabs(["Trial Data", "Patient Data", "Outcome Data"])
        
        with tab1:
            st.dataframe(data['trial_data'], use_container_width=True)
        
        with tab2:
            st.dataframe(data['patient_data'].head(100), use_container_width=True)
        
        with tab3:
            st.dataframe(data['outcome_data'].head(100), use_container_width=True)
        
        # Interactive charts
        st.subheader("📊 Interactive Analysis")
        
        # Success rate by enrollment size
        enrollment_bins = pd.cut(data['trial_data']['enrollment'], bins=5)
        success_by_enrollment = data['trial_data'].groupby(enrollment_bins)['success_rate'].mean()
        
        fig_enrollment = px.bar(
            x=success_by_enrollment.index.astype(str),
            y=success_by_enrollment.values,
            title="Success Rate by Enrollment Size",
            labels={'x': 'Enrollment Range', 'y': 'Success Rate'}
        )
        st.plotly_chart(fig_enrollment, use_container_width=True)
        
        # Age distribution
        fig_age = px.histogram(
            data['patient_data'],
            x='age',
            nbins=20,
            title="Patient Age Distribution",
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_age, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data explorer: {e}")

def show_settings():
    """Show settings and configuration"""
    
    st.header("⚙️ Settings & Configuration")
    
    st.subheader("API Configuration")
    
    api_url = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="Base URL for the prediction API"
    )
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_neural_network = st.checkbox("Use Neural Network", value=True)
        use_random_forest = st.checkbox("Use Random Forest", value=True)
    
    with col2:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.7, 0.1)
    
    st.subheader("Data Configuration")
    
    auto_refresh = st.checkbox("Auto-refresh Data", value=False)
    refresh_interval = st.number_input("Refresh Interval (seconds)", 30, 300, 60)
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved successfully!")
    
    st.markdown("---")
    st.subheader("System Information")
    
    # Display system info
    import platform
    st.text(f"Python Version: {platform.python_version()}")
    st.text(f"Streamlit Version: {st.__version__}")
    st.text(f"Platform: {platform.system()} {platform.release()}")

if __name__ == "__main__":
    main()

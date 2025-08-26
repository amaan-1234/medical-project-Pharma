"""
LLM utility functions for clinical trial protocol analysis
"""
import openai
import json
from typing import Dict, List, Optional, Tuple
import re
from pathlib import Path
import os
from config.config import LLM_CONFIG

class ProtocolAnalyzer:
    """
    Analyzes clinical trial protocols using LLM
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or LLM_CONFIG['api_key']
        if self.api_key:
            openai.api_key = self.api_key
        else:
            print("Warning: No OpenAI API key provided. Using fallback analysis.")
    
    def analyze_protocol(self, protocol_text: str) -> Dict:
        """
        Analyze clinical trial protocol using LLM
        
        Args:
            protocol_text: Raw protocol text
            
        Returns:
            Dictionary of extracted features and analysis
        """
        if not self.api_key:
            return self._fallback_analysis(protocol_text)
        
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(protocol_text)
            
            # Get LLM response
            response = openai.ChatCompletion.create(
                model=LLM_CONFIG['model'],
                messages=[
                    {"role": "system", "content": "You are a clinical trial expert. Analyze the protocol and extract key features."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=LLM_CONFIG['max_tokens'],
                temperature=LLM_CONFIG['temperature']
            )
            
            # Parse response
            analysis_text = response.choices[0].message.content
            return self._parse_llm_response(analysis_text)
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._fallback_analysis(protocol_text)
    
    def _create_analysis_prompt(self, protocol_text: str) -> str:
        """Create the prompt for LLM analysis"""
        return f"""
        Analyze the following clinical trial protocol and extract key features in JSON format:
        
        Protocol Text:
        {protocol_text[:2000]}...
        
        Please provide a JSON response with the following structure:
        {{
            "trial_phase": "Phase 1/2/3/4",
            "enrollment_target": "number or range",
            "duration_months": "number",
            "intervention_type": "Drug/Device/Procedure/Behavioral",
            "primary_outcome": "description",
            "inclusion_criteria": ["list of criteria"],
            "exclusion_criteria": ["list of criteria"],
            "randomization": "Yes/No",
            "blinding": "Single/Double/None",
            "placebo_control": "Yes/No",
            "multicenter": "Yes/No",
            "adaptive_design": "Yes/No",
            "interim_analysis": "Yes/No",
            "risk_factors": ["list of risk factors"],
            "success_probability": "High/Medium/Low",
            "confidence_score": "0.0-1.0"
        }}
        
        Focus on extracting objective, measurable features that could predict trial success.
        """
    
    def _parse_llm_response(self, response_text: str) -> Dict:
        """Parse LLM response and extract structured data"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return self._extract_features_from_text(response_text)
        except json.JSONDecodeError:
            return self._extract_features_from_text(response_text)
    
    def _extract_features_from_text(self, text: str) -> Dict:
        """Extract features from unstructured text response"""
        text_lower = text.lower()
        
        features = {
            "trial_phase": self._extract_phase(text_lower),
            "enrollment_target": self._extract_enrollment(text_lower),
            "duration_months": self._extract_duration(text_lower),
            "intervention_type": self._extract_intervention_type(text_lower),
            "primary_outcome": self._extract_primary_outcome(text_lower),
            "randomization": "Yes" if "random" in text_lower else "No",
            "blinding": self._extract_blinding(text_lower),
            "placebo_control": "Yes" if "placebo" in text_lower else "No",
            "multicenter": "Yes" if "multicenter" in text_lower else "No",
            "adaptive_design": "Yes" if "adaptive" in text_lower else "No",
            "interim_analysis": "Yes" if "interim" in text_lower else "No",
            "success_probability": self._extract_success_probability(text_lower),
            "confidence_score": 0.7  # Default confidence
        }
        
        return features
    
    def _extract_phase(self, text: str) -> str:
        """Extract trial phase from text"""
        if "phase 1" in text:
            return "Phase 1"
        elif "phase 2" in text:
            return "Phase 2"
        elif "phase 3" in text:
            return "Phase 3"
        elif "phase 4" in text:
            return "Phase 4"
        else:
            return "Unknown"
    
    def _extract_enrollment(self, text: str) -> str:
        """Extract enrollment target from text"""
        # Look for numbers followed by common enrollment terms
        enrollment_patterns = [
            r'(\d+)\s*patients?',
            r'enrollment.*?(\d+)',
            r'(\d+)\s*subjects?',
            r'(\d+)\s*participants?'
        ]
        
        for pattern in enrollment_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return "Unknown"
    
    def _extract_duration(self, text: str) -> str:
        """Extract trial duration from text"""
        duration_patterns = [
            r'(\d+)\s*months?',
            r'(\d+)\s*weeks?',
            r'(\d+)\s*years?'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return "Unknown"
    
    def _extract_intervention_type(self, text: str) -> str:
        """Extract intervention type from text"""
        if "drug" in text or "medication" in text:
            return "Drug"
        elif "device" in text:
            return "Device"
        elif "procedure" in text or "surgery" in text:
            return "Procedure"
        elif "behavioral" in text or "therapy" in text:
            return "Behavioral"
        else:
            return "Unknown"
    
    def _extract_primary_outcome(self, text: str) -> str:
        """Extract primary outcome from text"""
        outcome_keywords = {
            "survival": ["survival", "mortality", "death"],
            "quality_of_life": ["quality of life", "qol", "patient reported"],
            "disease_progression": ["progression", "disease free", "recurrence"],
            "safety": ["safety", "adverse", "toxicity"],
            "efficacy": ["efficacy", "response", "remission"]
        }
        
        for outcome, keywords in outcome_keywords.items():
            if any(keyword in text for keyword in keywords):
                return outcome.replace("_", " ").title()
        
        return "Unknown"
    
    def _extract_blinding(self, text: str) -> str:
        """Extract blinding information from text"""
        if "double blind" in text:
            return "Double"
        elif "single blind" in text:
            return "Single"
        elif "blind" in text:
            return "Yes"
        else:
            return "None"
    
    def _extract_success_probability(self, text: str) -> str:
        """Extract success probability assessment from text"""
        if any(word in text for word in ["high", "likely", "promising", "strong"]):
            return "High"
        elif any(word in text for word in ["medium", "moderate", "uncertain"]):
            return "Medium"
        elif any(word in text for word in ["low", "unlikely", "risky", "challenging"]):
            return "Low"
        else:
            return "Medium"
    
    def _fallback_analysis(self, protocol_text: str) -> Dict:
        """Fallback analysis when LLM is not available"""
        return self._extract_features_from_text(protocol_text)
    
    def calculate_risk_score(self, analysis: Dict) -> float:
        """
        Calculate risk score based on protocol analysis
        
        Args:
            analysis: Protocol analysis dictionary
            
        Returns:
            Risk score between 0 and 1 (higher = higher risk)
        """
        risk_score = 0.0
        
        # Phase risk (Phase 1 is highest risk)
        phase_risk = {"Phase 1": 0.8, "Phase 2": 0.6, "Phase 3": 0.4, "Phase 4": 0.2}
        if analysis.get("trial_phase") in phase_risk:
            risk_score += phase_risk[analysis["trial_phase"]] * 0.3
        
        # Design risk factors
        if analysis.get("randomization") == "No":
            risk_score += 0.1
        if analysis.get("blinding") == "None":
            risk_score += 0.1
        if analysis.get("placebo_control") == "No":
            risk_score += 0.05
        
        # Enrollment risk (smaller trials are riskier)
        try:
            enrollment = int(analysis.get("enrollment_target", "100"))
            if enrollment < 100:
                risk_score += 0.2
            elif enrollment < 500:
                risk_score += 0.1
        except ValueError:
            risk_score += 0.1
        
        # Duration risk (longer trials are riskier)
        try:
            duration = int(analysis.get("duration_months", "12"))
            if duration > 24:
                risk_score += 0.1
        except ValueError:
            pass
        
        # Normalize to 0-1 range
        return min(risk_score, 1.0)
    
    def generate_protocol_summary(self, analysis: Dict) -> str:
        """
        Generate human-readable protocol summary
        
        Args:
            analysis: Protocol analysis dictionary
            
        Returns:
            Formatted summary string
        """
        summary = f"""
        Clinical Trial Protocol Analysis Summary
        
        Trial Phase: {analysis.get('trial_phase', 'Unknown')}
        Enrollment Target: {analysis.get('enrollment_target', 'Unknown')} patients
        Duration: {analysis.get('duration_months', 'Unknown')} months
        Intervention Type: {analysis.get('intervention_type', 'Unknown')}
        
        Design Features:
        - Randomization: {analysis.get('randomization', 'Unknown')}
        - Blinding: {analysis.get('blinding', 'Unknown')}
        - Placebo Control: {analysis.get('placebo_control', 'Unknown')}
        - Multicenter: {analysis.get('multicenter', 'Unknown')}
        
        Risk Assessment:
        - Success Probability: {analysis.get('success_probability', 'Unknown')}
        - Risk Score: {self.calculate_risk_score(analysis):.2f}
        - Confidence: {analysis.get('confidence_score', 'Unknown')}
        """
        
        return summary.strip()

def create_sample_protocol() -> str:
    """
    Create a sample clinical trial protocol for testing
    
    Returns:
        Sample protocol text
    """
    return """
    Phase 2 Clinical Trial Protocol
    
    Study Title: Efficacy and Safety of Novel Drug XR-157 in Patients with Advanced Heart Failure
    
    Study Design: This is a multicenter, randomized, double-blind, placebo-controlled Phase 2 study to evaluate the efficacy and safety of XR-157 in patients with advanced heart failure.
    
    Primary Objective: To assess the effect of XR-157 on exercise capacity as measured by 6-minute walk distance at 12 weeks compared to placebo.
    
    Secondary Objectives: To evaluate safety, tolerability, and additional efficacy endpoints including quality of life measures and cardiac biomarkers.
    
    Study Population: Adult patients (18-80 years) with advanced heart failure (NYHA Class III-IV) despite optimal medical therapy.
    
    Inclusion Criteria:
    - Age 18-80 years
    - NYHA Class III or IV heart failure
    - Left ventricular ejection fraction ≤35%
    - Stable on optimal medical therapy for ≥4 weeks
    
    Exclusion Criteria:
    - Recent myocardial infarction (<3 months)
    - Unstable angina
    - Severe renal impairment (eGFR <30 mL/min/1.73m²)
    - Pregnancy or breastfeeding
    
    Study Treatment: Patients will be randomized 1:1 to receive XR-157 50 mg twice daily or matching placebo for 12 weeks.
    
    Sample Size: 200 patients (100 per treatment group)
    
    Study Duration: 12 weeks of treatment plus 4 weeks of follow-up
    
    Primary Endpoint: Change from baseline in 6-minute walk distance at Week 12
    
    Safety Assessments: Adverse events, vital signs, laboratory parameters, ECG, and echocardiography.
    
    Statistical Analysis: Primary analysis will use ANCOVA with baseline 6MWD as covariate. Sample size provides 80% power to detect a 30-meter difference between groups.
    
    Data Monitoring: An independent Data Monitoring Committee will review safety data every 6 months.
    """

if __name__ == "__main__":
    # Test the protocol analyzer
    analyzer = ProtocolAnalyzer()
    sample_protocol = create_sample_protocol()
    
    print("Analyzing sample protocol...")
    analysis = analyzer.analyze_protocol(sample_protocol)
    
    print("\nAnalysis Results:")
    print(json.dumps(analysis, indent=2))
    
    print(f"\nRisk Score: {analyzer.calculate_risk_score(analysis):.3f}")
    print(f"\nSummary:\n{analyzer.generate_protocol_summary(analysis)}")

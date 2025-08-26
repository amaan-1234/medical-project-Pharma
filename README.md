# AI-Powered Clinical Trial Outcome Predictor

## Problem Statement
Pharmaceutical companies spend billions of dollars on clinical trials with high failure rates. This project addresses this challenge by building a multi-modal AI system that predicts trial success by analyzing multiple data sources.

## Demo Link: https://amaan-project-pharma.streamlit.app/

## Solution Overview
Our system combines:
- **Deep Learning Models**: For patient demographics and biomarker analysis
- **Large Language Models (LLMs)**: For clinical trial protocol analysis
- **Historical Data Analysis**: From public registries like ClinicalTrials.gov

## Project Structure
```
medical-project/
├── data/                   # Data storage and sample datasets
├── models/                 # ML model definitions and training
├── api/                    # FastAPI backend
├── web/                    # Streamlit web interface
├── utils/                  # Utility functions and helpers
├── notebooks/              # Jupyter notebooks for exploration
├── config/                 # Configuration files
└── tests/                  # Unit tests
```

## Key Features
- **Multi-modal Analysis**: Combines text, tabular, and genomic data
- **Real-time Predictions**: API endpoints for instant trial outcome predictions
- **Interactive Web Interface**: User-friendly dashboard for data exploration
- **Comprehensive Reporting**: Detailed analysis and visualization of results

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd medical-project
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

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### 1. Start the API Backend
```bash
python api/main.py
```
The API will be available at `http://localhost:8000`

### 2. Launch the Web Interface
```bash
streamlit run web/app.py
```
The web app will open at `http://localhost:8501`

### 3. Run Model Training
```bash
python models/train.py
```

## API Endpoints

- `POST /predict/trial`: Predict trial outcome from protocol and patient data
- `GET /trials/history`: Get historical trial data
- `POST /analyze/protocol`: Analyze clinical trial protocol text
- `GET /models/status`: Check model health and performance

## Model Architecture

### Deep Learning Component
- **Patient Data Model**: Multi-layer perceptron for demographic and biomarker features
- **Genomic Model**: CNN-based architecture for genetic sequence analysis
- **Ensemble Model**: Combines predictions from multiple specialized models

### LLM Component
- **Protocol Analyzer**: Uses GPT-4 or similar for protocol text analysis
- **Feature Extractor**: Extracts key trial characteristics from unstructured text
- **Risk Assessment**: Evaluates protocol risk factors

## Data Sources
- **ClinicalTrials.gov**: Public trial registry data
- **Patient Demographics**: Age, gender, medical history
- **Biomarkers**: Lab values, genetic markers, vital signs
- **Protocol Documents**: Trial design, inclusion/exclusion criteria

## Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision/Recall**: For positive and negative outcomes
- **AUC-ROC**: Model discrimination ability
- **F1-Score**: Balanced performance measure

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Contact
For questions or support, please open an issue on GitHub.

# CodeShield

CodeShield is a Streamlit-based student submission similarity analyzer that helps educators identify potential plagiarism patterns for human review. It is an explainable, ML-based system designed to flag suspiciously similar assignments. It assists instructors by highlighting areas of concern; it never accuses students automatically.

## Overview

CodeShield analyzes student submissions using multiple similarity metrics (lexical, semantic, and structural) to detect paraphrasing, copying, and text overlaps. It uses a machine learning model (XGBoost) to predict similarity probability, but falls back to heuristic scoring when a model is not trained. It explains its reasoning with interactive graphs and highlighted matching phrases.

### Key Features
- **Multi-document Upload**: Support for PDF, DOCX, TXT, and MD files.
- **Machine Learning Powered**: Uses an XGBoost classifier trained on semantic, lexical, and structural features.
- **Explainable Results**: Provides pair-wise detailed inspection, including matching phrase highlights, a feature breakdown chart, and reasons for flagging.
- **Interactive Visualizations**: View a similarity heatmap across the entire batch, and a network graph of flagged pairs.
- **Privacy First**: Everything runs locally. Your data does not leave your machine.

## Project Structure

```
CodeShield/
├── src/
│   ├── extractors/                 # Text extraction modules (PDF, DOCX, TXT)
│   ├── similarity/                 # Lexical, semantic, and structural similarity logic
│   ├── ml/                         # ML feature extraction and model prediction
│   └── visuals/                    # Plotly-based visualizations (Heatmap, Network)
├── training/                       # Scripts to generate synthetic pairs and train the model
├── data/                           # Data directory for training docs and demo submissions
├── models/                         # Saved ML models (e.g., XGBoost `.pkl` files)
├── requirements.txt                # Python dependencies
└── streamlit_exe.ipynb             # Main execution point / Jupyter integration
```

## Getting Started

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Installation

1. **Clone the repository** (if applicable) and navigate to the project root directory.

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   
   # Windows:
   .venv\Scripts\activate
   
   # Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Generating Training Data & Training the Model

If you want to train the model from scratch (using synthetic dummy files):
1. **Generate synthetic pairs**: 
   ```bash
   python training/generate_synthetic_pairs.py
   ```
2. **Train the model**: 
   ```bash
   python training/train_model.py
   ```

### Running the Application

To start the Streamlit application, typically you would run:
```bash
streamlit run app.py
```
*(Note: Refer to `streamlit_exe.ipynb` or runner scripts if you are using Jupyter or a custom runner environment to launch the UI.)*

## Similarity Scoring Metrics

CodeShield extracts the following features to detect similarity:
1. **Lexical Similarity**: TF-IDF cosine similarity.
2. **Semantic Similarity**: Sentence transformer embeddings.
3. **Structural Similarity**: General document structure and layout.
4. **Word & Phrase Overlap**: Exact multi-word matching sequences.

These features feed into the trained model (or a heuristic weighted average fallback) to output a confidence score that instructors can review.

## Future Roadmap

- **Code File Support**: Direct syntax and AST parsing for programming languages (Python, Java, C++, etc.).
- **Batch Processing**: Allow folder uploads for processing large classroom sets.
- **Database Integration**: SQLite/Postgres integration for tracking submission history across multiple assignments.
- **API Endpoints**: REST API (Flask/FastAPI) for LMS integration (Canvas, Moodle, etc.).

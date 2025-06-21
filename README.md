# 🫀 Heart Disease Data Mining Platform

A comprehensive interactive dashboard for heart disease data analysis, preprocessing, and machine learning modeling using Streamlit.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Troubleshooting](#troubleshooting)

## Overview

This project provides an interactive web-based platform for analyzing heart disease data using various data mining and machine learning techniques. The application offers comprehensive data preprocessing, quality assessment, exploratory data analysis, and automated model training capabilities.

## Features

- **Interactive Dashboard**: Netflix-styled UI built with Streamlit
- **Data Quality Assessment**: Comprehensive data validation and quality metrics
- **Exploratory Data Analysis**: Interactive visualizations and statistical analysis
- **Data Preprocessing Pipeline**: Multiple preprocessing options including:
  - Null value handling (intelligent imputation, row removal)
  - Outlier detection and handling (capping, removal)
  - Categorical encoding (one-hot, label encoding)
- **Automated ML Pipeline**: Tests all combinations of preprocessing methods with multiple models
- **Model Training & Evaluation**: Support for SVM, Random Forest, and XGBoost
- **Feature Analysis**: Correlation analysis and feature importance visualization
- **Portfolio Showcase**: Demonstrates various data mining techniques

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone or download the project** to your local machine

2. **Navigate to the project directory**:
   ```powershell
   cd "c:\Users\shahr\OneDrive\Desktop\Assignments\SEM 6\CSC4600\Heart Disease Analysis"
   ```

3. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv heart_disease_env
   ```

4. **Activate the virtual environment**:
   ```powershell
   heart_disease_env\Scripts\Activate.ps1
   ```
   
   *Note: If you encounter an execution policy error, run:*
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

5. **Install required packages**:
   ```powershell
   pip install -r requirements.txt
   ```

## Running the Application

1. **Ensure you're in the project directory and virtual environment is activated**

2. **Start the Streamlit application**:
   ```powershell
   streamlit run app.py
   ```

3. **Access the application**:
   - The application will automatically open in your default web browser
   - If it doesn't open automatically, navigate to: `http://localhost:8501`

4. **Stop the application**:
   - Press `Ctrl + C` in the terminal/PowerShell window

## Project Structure

```
Heart Disease Analysis/
├── app.py                      # Main Streamlit application
├── automated_pipeline.py       # Automated ML pipeline implementation
├── data_preprocessing.py       # Data preprocessing utilities
├── data_quality.py            # Data quality assessment tools
├── visualization_utils.py     # Visualization helper functions
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── data/
│   └── heart_disease_uci.csv # Heart disease dataset
└── __pycache__/              # Python cache files
```

## Usage Guide

### Navigation
The application consists of 5 main tabs:

1. **Data Overview**: 
   - Upload and view dataset
   - Basic statistics and data info
   - Sample data preview

2. **Exploratory Analysis**:
   - Interactive visualizations
   - Statistical summaries
   - Correlation analysis

3. **Data Quality**:
   - Missing values assessment
   - Data type validation
   - Domain rule checking
   - Outlier detection

4. **Automated Pipeline**:
   - Configure preprocessing options
   - Run automated model testing
   - Compare results across different combinations

5. **Portfolio Showcase**:
   - Demonstrates various data mining techniques
   - Feature engineering examples
   - Advanced visualizations

### Getting Started
1. Launch the application using the instructions above
2. The dataset is automatically loaded from `data/heart_disease_uci.csv`
3. Navigate through the tabs to explore different features
4. Use the sidebar controls to customize analysis parameters

## Dataset

The application uses the **Heart Disease UCI dataset** which contains:
- **921 instances** of patient data
- **15 features** including demographic, clinical, and test results
- **Target variable**: Heart disease presence (0 = no disease, 1+ = disease present)

### Key Features:
- Age, sex, chest pain type
- Resting blood pressure, cholesterol levels
- ECG results, exercise-induced angina
- And more clinical indicators

## Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost
- **Statistical Analysis**: SciPy

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Streamlit command not recognized**:
   - Ensure virtual environment is activated
   - Try: `python -m streamlit run app.py`

3. **Port already in use**:
   ```powershell
   streamlit run app.py --server.port 8502
   ```

4. **Data file not found**:
   - Ensure `heart_disease_uci.csv` is in the `data/` folder
   - Check file path in `app.py` if needed

5. **Performance issues**:
   - Close other browser tabs
   - Restart the Streamlit server
   - Check available system memory

### Getting Help

If you encounter issues:
1. Check the terminal/PowerShell for error messages
2. Ensure all dependencies are installed correctly
3. Verify Python version compatibility (3.8+)
4. Try restarting the application

## Notes

- The application loads data automatically on startup
- All preprocessing and model training is done in real-time
- Results are not saved between sessions

---

**Built for CSC4600 - Data Mining Course**  
*Comprehensive heart disease analysis and machine learning platform*

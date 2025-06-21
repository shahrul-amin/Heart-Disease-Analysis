"""
Heart Disease Data Mining - Interactive Dashboard
Comprehensive data preprocessing and visualization platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Import custom modules
from data_preprocessing import HeartDiseasePreprocessor
from data_quality import DataQualityAssessor
from visualization_utils import HeartDiseaseVisualizer
from automated_pipeline import AutomatedPipeline

# Add scikit-learn and xgboost imports for modeling
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import json
import os

def clean_dataframe_for_plotting(df, numeric_cols=None):
    """Clean dataframe by handling NaN values for plotting"""
    df_clean = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in df_clean.columns:
            if df_clean[col].isnull().any():
                # Fill NaN with median for numeric columns
                median_val = df_clean[col].median()
                if pd.isna(median_val):  # If median is also NaN, use 0
                    median_val = 0
                df_clean[col] = df_clean[col].fillna(median_val)
    
    # Also handle categorical columns for consistency
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in df_clean.columns:
            if df_clean[col].isnull().any():
                # Fill NaN with mode for categorical columns, or 'Unknown' if no mode
                mode_val = df_clean[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                df_clean[col] = df_clean[col].fillna(fill_val)
    
    return df_clean

# Page configuration
st.set_page_config(
    page_title="Heart Disease Data Mining Platform",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Netflix-inspired theme styling
st.markdown("""
<style>
    /* Netflix-style background - pure black with subtle red accents */
    .stApp {
        background: #000000;
        color: #ffffff;
    }
    
    /* Sidebar styling with Netflix theme */
    .css-1d391kg {
        background: #000000;
        border-right: 2px solid #e50914;
    }
    
    /* Main content area with Netflix styling */
    .main .block-container {
        background: rgba(0, 0, 0, 0.95);
        border-radius: 8px;
        padding: 2rem;
        border: 1px solid rgba(229, 9, 20, 0.3);
        box-shadow: 0 4px 20px rgba(229, 9, 20, 0.1);
    }    
    .main-header {
        font-size: 2.5rem;
        color: #e50914;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        border-bottom: 3px solid #e50914;
        padding-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    .sub-header {
        font-size: 1.4rem;
        color: #ffffff;
        margin-bottom: 1rem;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #141414 0%, #e50914 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.3);
        transition: transform 0.2s ease;
        border: 1px solid rgba(229, 9, 20, 0.4);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.5);
        border: 1px solid #e50914;
    }    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 0.5rem;
        color: #ffffff;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    
    /* Sidebar styling - Netflix dark theme */
    .sidebar .sidebar-content {
        background: #000000;
    }
    
    /* Input fields and selectboxes - Netflix style */
    .stSelectbox > div > div {
        background-color: #141414;
        border: 2px solid #e50914;
        border-radius: 4px;
        color: #ffffff;
    }
    
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    /* Buttons - Netflix red */
    .stButton > button {
        background: #e50914;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.6rem 2rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(229, 9, 20, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: #f40612;
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.6);
    }    
    /* Methodology box - Netflix card style */
    .methodology-box {
        background: #141414;
        border: 1px solid #e50914;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.8);
        color: #ffffff;
    }
    
    /* Status indicators with Netflix theme */
    .status-indicator {
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem 0;
    }
    
    .status-success {
        background-color: #46d369;
        color: #000000;
        border: 1px solid #46d369;
    }
    
    .status-warning {
        background-color: #f9c23c;
        color: #000000;
        border: 1px solid #f9c23c;
    }
    
    .status-error {
        background-color: #e50914;
        color: #ffffff;
        border: 1px solid #e50914;
    }
    
    /* Text and labels */
    .stMarkdown, .stText {
        color: #ffffff;
    }
    
    /* Tabs styling - Netflix style */
    .stTabs > div > div > div {
        background-color: #000000;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #141414;
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 4px;
        margin: 0 2px;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #e50914;
        color: #ffffff;
        border: 1px solid #e50914;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #333333;
        border: 1px solid #e50914;
    }    
    /* Dataframe styling - Netflix dark theme */
    .stDataFrame {
        background-color: #141414;
        border: 1px solid #333333;
        border-radius: 4px;
    }
    
    /* Plots and charts background */
    .js-plotly-plot {
        background-color: #141414 !important;
        border-radius: 4px;
        border: 1px solid #333333;
    }
    
    /* Additional form elements styling */
    .stTextInput > div > div > input {
        background-color: #141414;
        border: 2px solid #333333;
        border-radius: 4px;
        color: #ffffff;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #e50914;
    }
    
    .stNumberInput > div > div > input {
        background-color: #141414;
        border: 2px solid #333333;
        border-radius: 4px;
        color: #ffffff;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #e50914;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #141414;
        border: 2px solid #333333;
        border-radius: 4px;
        color: #ffffff;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #e50914;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background-color: #141414;
        border: 2px dashed #333333;
        border-radius: 4px;
    }
    
    .stFileUploader:hover > div {
        border-color: #e50914;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #e50914;
    }
    
    /* Checkbox styling */
    .stCheckbox > label > div {
        background-color: #141414;
        border: 2px solid #333333;
    }
    
    .stCheckbox > label > div[data-checked="true"] {
        background-color: #e50914;
        border-color: #e50914;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: #141414;
        border-radius: 4px;
        padding: 0.5rem;
        border: 1px solid #333333;
    }
    
    /* Success/Info/Warning/Error message styling */
    .stAlert > div {
        border-radius: 4px;
        border-left: 4px solid #e50914;
        background-color: #141414;
        color: #ffffff;
    }
    
    /* Metric styling */
    .metric-container {
        background: #141414;
        border-radius: 4px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #333333;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #e50914;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #141414;
        border: 1px solid #333333;
        border-radius: 4px;
        color: #ffffff;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #e50914;
    }
    
    /* Code block styling */
    .stCodeBlock {
        background-color: #141414;
        border: 1px solid #333333;
        border-radius: 4px;
    }
    
    /* Table styling */
    .dataframe {
        background-color: #141414;
        color: #ffffff;
    }
    
    .dataframe th {
        background-color: #e50914;
        color: #ffffff;
        font-weight: 700;
    }
    
    .dataframe td {
        background-color: #141414;
        color: #ffffff;
        border: 1px solid #333333;
    }
    
    .dataframe tr:hover {
        background-color: #1f1f1f;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #e50914;
    }
    
    /* Custom Netflix-style scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #000000;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #e50914;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #f40612;
    }
    
    /* Recommendation Cards Styling */
    .recommendation-card {
        background: linear-gradient(135deg, #141414 0%, #1f1f1f 100%);
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.3);
        border-color: #e50914;
    }
    
    .recommendation-card:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--priority-color);
    }
    
    .recommendation-high {
        --priority-color: #e50914;
    }
    
    .recommendation-medium {
        --priority-color: #f9c23c;
    }
    
    .recommendation-low {
        --priority-color: #46d369;
    }
    
    .recommendation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 1px solid #333333;
        padding-bottom: 0.5rem;
    }
    
    .priority-badge {
        background: var(--priority-color);
        color: #ffffff;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .category-tag {
        background: #333333;
        color: #ffffff;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .recommendation-content {
        color: #ffffff;
        line-height: 1.6;
    }
    
    .recommendation-issue {
        background: rgba(229, 9, 20, 0.1);
        border: 1px solid rgba(229, 9, 20, 0.3);
        border-radius: 4px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    
    .recommendation-solution {
        background: rgba(70, 211, 105, 0.1);
        border: 1px solid rgba(70, 211, 105, 0.3);
        border-radius: 4px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    
    .recommendation-icon {
        font-size: 1.2rem;
        margin-right: 0.5rem;
    }
    
    .business-recommendation-card {
        background: linear-gradient(135deg, #141414 0%, #1f1f1f 100%);
        border: 1px solid #e50914;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 3px 12px rgba(229, 9, 20, 0.2);
        transition: all 0.3s ease;
    }
    
    .business-recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.4);
    }
    
    .business-rec-header {
        color: #e50914;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }
    
    .business-rec-content {
        color: #ffffff;
        margin-left: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'preprocessing_complete' not in st.session_state:
    st.session_state.preprocessing_complete = False

def load_data():
    """Load the heart disease dataset"""
    try:
        data_path = "data/heart_disease_uci.csv"
        df = pd.read_csv(data_path, index_col='id')
        st.session_state.original_data = df
        st.session_state.data_loaded = True
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'heart_disease_uci.csv' is in the 'data' directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Auto-load data on startup
if not st.session_state.data_loaded:
    df = load_data()

def create_methodology_section():
    """Create methodology documentation section"""
    st.markdown('<div class="methodology-box">', unsafe_allow_html=True)
    st.markdown("### Data Preprocessing Methodology")
    
    methodology_steps = [
        {
            "step": "1. Data Loading & Initial Assessment",
            "description": "Load dataset and perform initial quality assessment including shape, data types, and basic statistics.",
            "justification": "Essential for understanding data structure and identifying immediate quality issues."
        },
        {
            "step": "2. Missing Value Analysis",
            "description": "Identify missing value patterns and apply appropriate imputation strategies based on data distribution.",
            "justification": "Prevents bias and ensures complete datasets for analysis. Method selection based on statistical properties."
        },
        {
            "step": "3. Outlier Detection & Treatment",
            "description": "Use IQR method and z-score analysis to identify outliers, then apply capping, removal, or transformation.",
            "justification": "Outliers can significantly impact model performance and statistical analysis results."
        },
        {
            "step": "4. Data Type Standardization",
            "description": "Convert boolean text to integers, ensure proper categorical and numeric types.",
            "justification": "Consistent data types enable proper statistical analysis and model training."
        },
        {
            "step": "5. Feature Engineering",
            "description": "Create derived features like age groups, cholesterol categories, and binary target variables.",
            "justification": "Domain-specific features can improve model interpretability and performance."
        },
        {
            "step": "6. Categorical Encoding",
            "description": "Apply one-hot encoding or label encoding based on the nature of categorical variables.",
            "justification": "Machine learning algorithms require numerical inputs for categorical variables."
        }
    ]
    
    for step in methodology_steps:
        with st.expander(f"{step['step']}"):
            st.write(f"**Description:** {step['description']}")
            st.write(f"**Justification:** {step['justification']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_data_overview_page():
    """Create data overview page"""
    st.markdown('<h1 class="main-header">Heart Disease Data Mining Platform</h1>', unsafe_allow_html=True)
    
    # Clear sidebar content for this page
    st.sidebar.empty()
    
    if st.session_state.data_loaded:
        df = st.session_state.original_data
        
        # Dataset overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Records</div>
                <div class="metric-value">{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Features</div>
                <div class="metric-value">{len(df.columns)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Missing Data</div>
                <div class="metric-value">{missing_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            heart_disease_rate = (df['num'] > 0).mean() * 100 if 'num' in df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Disease Rate</div>
                <div class="metric-value">{heart_disease_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Interactive dataset preview
        st.markdown('<h2 class="sub-header">Dataset Preview</h2>', unsafe_allow_html=True)
        
        # Add search and filter functionality
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Search in dataset:", placeholder="Enter search term...")
        with col2:
            show_rows = st.selectbox("Rows to display:", [10, 25, 50, 100], index=0)
        
        # Filter dataframe based on search
        display_df = df.head(show_rows)
        if search_term:
            # Search across all string columns
            string_cols = df.select_dtypes(include=['object']).columns
            mask = df[string_cols].astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            display_df = df[mask].head(show_rows)
        
        st.dataframe(
            display_df, 
            use_container_width=True,
            hide_index=False,
            column_config={
                col: st.column_config.NumberColumn(
                    format="%.2f" if df[col].dtype in ['float64', 'float32'] else "%d"
                ) for col in df.select_dtypes(include=[np.number]).columns
            }
        )
        
        # Interactive statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="sub-header">Numeric Features Statistics</h3>', unsafe_allow_html=True)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            stats_df = df[numeric_cols].describe().round(2)
            st.dataframe(
                stats_df, 
                use_container_width=True,
                column_config={
                    col: st.column_config.NumberColumn(format="%.2f") 
                    for col in stats_df.columns
                }
            )
        
        with col2:
            st.markdown('<h3 class="sub-header">Categorical Features Info</h3>', unsafe_allow_html=True)
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                cat_info = pd.DataFrame({
                    'Feature': cat_cols,
                    'Unique Values': [df[col].nunique() for col in cat_cols],
                    'Most Common': [df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A' for col in cat_cols]
                })
                st.dataframe(cat_info, use_container_width=True)
            else:
                st.info("No categorical features found in the dataset.")
        
        # Interactive feature descriptions
        st.markdown('<h2 class="sub-header">Feature Descriptions</h2>', unsafe_allow_html=True)
        
        feature_descriptions = {
            'age': 'Age of the patient in years',
            'sex': 'Gender of the patient (Male/Female)',
            'cp': 'Chest pain type (typical angina, atypical angina, non-anginal pain, asymptomatic)',
            'trestbps': 'Resting blood pressure in mm Hg',
            'chol': 'Serum cholesterol in mg/dl',
            'fbs': 'Fasting blood sugar > 120 mg/dl (TRUE/FALSE)',
            'restecg': 'Resting electrocardiographic results',
            'thalch': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (TRUE/FALSE)',
            'oldpeak': 'ST depression induced by exercise relative to rest',
            'slope': 'Slope of the peak exercise ST segment',
            'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
            'thal': 'Thalassemia type (normal, fixed defect, reversible defect)',
            'num': 'Target variable: Heart disease diagnosis (0-4, 0=no disease, 1-4=disease severity)'
        }
        
        desc_df = pd.DataFrame(list(feature_descriptions.items()), columns=['Feature', 'Description'])
        
        # Add search functionality for feature descriptions
        search_feature = st.text_input("Search features:", placeholder="Enter feature name...")
        if search_feature:
            mask = desc_df['Feature'].str.contains(search_feature, case=False, na=False) | \
                   desc_df['Description'].str.contains(search_feature, case=False, na=False)
            desc_df = desc_df[mask]
        
        st.dataframe(
            desc_df, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn("Feature", width="medium"),
                "Description": st.column_config.TextColumn("Description", width="large")
            }
        )
    else:
        st.error("Failed to load dataset. Please check if 'heart_disease_uci.csv' exists in the 'data' directory.")

def create_exploratory_analysis_page():
    """Create exploratory data analysis page"""
    st.markdown('<h1 class="main-header">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Data Overview page.")
        return
    
    df = st.session_state.original_data
    visualizer = HeartDiseaseVisualizer()
    
    # Interactive filters
    st.sidebar.markdown("### Analysis Filters")
    
    # Age filter
    if 'age' in df.columns:
        age_range = st.sidebar.slider(
            "Age Range",
            min_value=int(df['age'].min()),
            max_value=int(df['age'].max()),
            value=(int(df['age'].min()), int(df['age'].max()))
        )
        df_filtered = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    else:
        df_filtered = df
    
    # Gender filter
    if 'sex' in df.columns:
        gender_options = df['sex'].unique().tolist()
        selected_genders = st.sidebar.multiselect(
            "Gender",
            options=gender_options,
            default=gender_options
        )
        df_filtered = df_filtered[df_filtered['sex'].isin(selected_genders)]
    
    # Dataset filter
    if 'dataset' in df.columns:
        dataset_options = df['dataset'].unique().tolist()
        selected_datasets = st.sidebar.multiselect(
            "Dataset Source",
            options=dataset_options,
            default=dataset_options
        )
        df_filtered = df_filtered[df_filtered['dataset'].isin(selected_datasets)]
    
    st.info(f"Filtered dataset contains {len(df_filtered)} records out of {len(df)} total records.")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview Dashboard", "Correlations", "Distributions", "Target Analysis"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Comprehensive Overview</h2>', unsafe_allow_html=True)
        overview_fig = visualizer.create_overview_dashboard(df_filtered)
        st.plotly_chart(overview_fig, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Feature Correlations</h2>', unsafe_allow_html=True)
        
        # Correlation heatmap
        corr_fig = visualizer.create_correlation_heatmap(df_filtered)
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Top correlations with target
        if 'num' in df_filtered.columns:
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
            target_corr = df_filtered[numeric_cols].corr()['num'].abs().sort_values(ascending=False)[1:6]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Top Correlations with Heart Disease")
                for feature, corr_value in target_corr.items():
                    st.write(f"**{feature}**: {corr_value:.3f}")
            
            with col2:
                st.markdown("#### Correlation Insights")
                st.write("- Strong correlations (>0.5) may indicate redundant features")
                st.write("- Moderate correlations (0.3-0.5) often provide valuable insights")
                st.write("- Weak correlations (<0.3) might still be useful in combination")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Feature Distributions</h2>', unsafe_allow_html=True)
        
        # Select columns for distribution analysis
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect(
            "Select features for distribution analysis:",
            options=numeric_cols,
            default=numeric_cols[:4]
        )
        
        if selected_cols:
            # Distribution plots
            dist_fig = visualizer.create_distribution_plots(df_filtered, selected_cols)
            st.pyplot(dist_fig)
            plt.close()
            
            # Outlier analysis
            st.markdown("#### Outlier Detection")
            outlier_fig = visualizer.create_outlier_analysis(df_filtered, selected_cols)
            st.plotly_chart(outlier_fig, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="sub-header">Target Variable Analysis</h2>', unsafe_allow_html=True)
        
        if 'num' in df_filtered.columns:
            target_fig = visualizer.create_target_analysis(df_filtered, 'num')
            st.plotly_chart(target_fig, use_container_width=True)
            
            # Target statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Target Distribution")
                target_counts = df_filtered['num'].value_counts().sort_index()
                for value, count in target_counts.items():
                    percentage = (count / len(df_filtered)) * 100
                    st.write(f"**Level {value}**: {count} patients ({percentage:.1f}%)")
            
            with col2:
                st.markdown("#### Clinical Interpretation")
                st.write("- **0**: No heart disease")
                st.write("- **1**: Mild heart disease")
                st.write("- **2**: Moderate heart disease")
                st.write("- **3**: Severe heart disease")
                st.write("- **4**: Very severe heart disease")

def create_preprocessing_page():
    """Create data preprocessing page"""
    st.markdown('<h1 class="main-header">Data Preprocessing Pipeline</h1>', unsafe_allow_html=True)
    
    # Clear sidebar content for this page
    st.sidebar.empty()
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Data Overview page.")
        return
    
    # Methodology section
    create_methodology_section()
    
    # Preprocessing configuration
    st.markdown('<h2 class="sub-header">Preprocessing Configuration</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        missing_strategy = st.selectbox(
            "Missing Value Strategy",
            options=['intelligent', 'drop', 'simple'],
            help="Intelligent: Uses statistical properties to determine best imputation method"
        )
    
    with col2:
        outlier_method = st.selectbox(
            "Outlier Handling",
            options=['cap', 'remove', 'transform'],
            help="Cap: Limit outliers to reasonable bounds"
        )
    
    with col3:
        encoding_method = st.selectbox(
            "Categorical Encoding",
            options=['onehot', 'label'],
            help="One-hot: Creates binary columns for each category"
        )
    
    # Run preprocessing
    if st.button("Run Preprocessing Pipeline", type="primary"):
        with st.spinner("Running preprocessing pipeline..."):
            preprocessor = HeartDiseasePreprocessor()
            
            result = preprocessor.run_full_pipeline(
                file_path="data/heart_disease_uci.csv",
                missing_strategy=missing_strategy,
                outlier_method=outlier_method,
                encoding_method=encoding_method
            )
            
            if result is not None:
                processed_data, pipeline_report = result
                
                # Store in session state
                st.session_state.processed_data = processed_data
                st.session_state.pipeline_report = pipeline_report
                st.session_state.preprocessing_complete = True
                
                st.success("Preprocessing completed successfully!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Before Preprocessing")
                    st.write(f"Shape: {pipeline_report['original_shape']}")
                    st.write(f"Missing values: {st.session_state.original_data.isnull().sum().sum()}")
                
                with col2:
                    st.markdown("#### After Preprocessing")
                    st.write(f"Shape: {pipeline_report['final_shape']}")
                    st.write(f"Missing values: {processed_data.isnull().sum().sum()}")
                
                # Processing log
                st.markdown('<h3 class="sub-header">Processing Log</h3>', unsafe_allow_html=True)
                log_df = pd.DataFrame(pipeline_report['processing_log'])
                st.dataframe(log_df, use_container_width=True)
    
    # Show comparison if preprocessing is complete
    if st.session_state.preprocessing_complete:
        st.markdown('<h2 class="sub-header">Before vs After Comparison</h2>', unsafe_allow_html=True)
        
        visualizer = HeartDiseaseVisualizer()
        comparison_fig = visualizer.create_preprocessing_comparison(
            st.session_state.original_data,
            st.session_state.processed_data
        )
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Processed data preview
        st.markdown('<h3 class="sub-header">Processed Dataset Preview</h3>', unsafe_allow_html=True)
        st.dataframe(st.session_state.processed_data.head(10), use_container_width=True)
        
        # Download processed data
        csv_buffer = BytesIO()
        st.session_state.processed_data.to_csv(csv_buffer, index=True)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="Download Processed Dataset",
            data=csv_data,
            file_name="heart_disease_processed.csv",
            mime="text/csv"
        )

def create_quality_assessment_page():
    """Create data quality assessment page"""
    st.markdown('<h1 class="main-header">Data Quality Assessment</h1>', unsafe_allow_html=True)
    
    # Clear sidebar content for this page
    st.sidebar.empty()
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Data Overview page.")
        return
    
    df = st.session_state.original_data
    
    # Run quality assessment
    if st.button("Run Quality Assessment", type="primary"):
        with st.spinner("Assessing data quality..."):
            assessor = DataQualityAssessor()
            quality_report = assessor.run_full_assessment(df)
            st.session_state.quality_report = quality_report
    
    if 'quality_report' in st.session_state:
        quality_report = st.session_state.quality_report
        
        # Overall quality score
        if 'overall_score' in quality_report:
            overall_score = quality_report['overall_score']['overall']
            grade = quality_report['overall_score']['grade']
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 1rem; color: white;">
                    <h2>Overall Data Quality Score</h2>
                    <div style="font-size: 3rem; font-weight: bold;">{overall_score:.1f}/100</div>
                    <div style="font-size: 2rem; margin-top: 1rem;">Grade: {grade}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Quality dashboard
        st.markdown('<h2 class="sub-header">Quality Assessment Dashboard</h2>', unsafe_allow_html=True)
        visualizer = HeartDiseaseVisualizer()
        quality_fig = visualizer.create_data_quality_dashboard(quality_report)
        st.plotly_chart(quality_fig, use_container_width=True)        # Component scores
        if 'overall_score' in quality_report:
            st.markdown('<h3 class="sub-header">Component Scores</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            component_scores = quality_report['overall_score']['component_scores']
            
            # Convert weighted scores back to individual scores
            with col1:
                completeness_score = component_scores.get('completeness', 0) / 0.4
                st.metric("Completeness", f"{completeness_score:.1f}")
            
            with col2:
                validity_score = component_scores.get('validity', 0) / 0.3
                st.metric("Validity", f"{validity_score:.1f}")
            
            with col3:
                consistency_score = component_scores.get('consistency', 0) / 0.2
                st.metric("Consistency", f"{consistency_score:.1f}")
            
            with col4:
                accuracy_score = component_scores.get('accuracy', 0) / 0.1
                st.metric("Accuracy", f"{accuracy_score:.1f}")
        
        # Detailed assessment tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Completeness", "Validity", "Consistency", "Accuracy"])
        
        with tab1:
            st.markdown('<h3 class="sub-header">Completeness Assessment</h3>', unsafe_allow_html=True)
            
            if 'completeness' in quality_report:
                completeness = quality_report['completeness']
                
                # Missing values by column
                missing_data = []
                for col, info in completeness['columns'].items():
                    if info['missing_count'] > 0:
                        missing_data.append({
                            'Column': col,
                            'Missing Count': info['missing_count'],
                            'Missing %': f"{info['missing_rate']:.2f}%",
                            'Completeness %': f"{info['completeness_rate']:.2f}%"
                        })
                
                if missing_data:
                    st.dataframe(pd.DataFrame(missing_data), use_container_width=True)
                else:
                    st.success("No missing values found in the dataset!")
        
        with tab2:
            st.markdown('<h3 class="sub-header">Validity Assessment</h3>', unsafe_allow_html=True)
            
            if 'validity' in quality_report:
                validity = quality_report['validity']
                
                # Domain rule violations
                if validity['domain_rules']:
                    st.markdown("#### Domain Rule Violations")
                    for col, violations in validity['domain_rules'].items():
                        st.error(f"**{col}**: {', '.join(violations)}")
                else:
                    st.success("No domain rule violations found!")
                
                # Outliers
                st.markdown("#### Outlier Summary")
                outlier_data = []
                for col, info in validity['outliers'].items():
                    outlier_data.append({
                        'Feature': col,
                        'Outlier Count': info['count'],
                        'Outlier %': f"{info['percentage']:.2f}%",
                        'Lower Bound': f"{info['bounds'][0]:.2f}",
                        'Upper Bound': f"{info['bounds'][1]:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(outlier_data), use_container_width=True)
        
        with tab3:
            st.markdown('<h3 class="sub-header">Consistency Assessment</h3>', unsafe_allow_html=True)
            
            if 'consistency' in quality_report:
                consistency = quality_report['consistency']
                
                # Duplicates
                dup_count = consistency['duplicates']['count']
                dup_pct = consistency['duplicates']['percentage']
                
                if dup_count > 0:
                    st.warning(f"Found {dup_count} duplicate records ({dup_pct:.2f}%)")
                else:
                    st.success("No duplicate records found!")
                
                # Format issues
                if consistency['formats']:
                    st.markdown("#### Format Issues")
                    for col, issue in consistency['formats'].items():
                        st.warning(f"**{col}**: {issue}")
                else:
                    st.success("No format inconsistencies found!")
        
        with tab4:
            st.markdown('<h3 class="sub-header">Accuracy Assessment</h3>', unsafe_allow_html=True)
            
            if 'accuracy' in quality_report:
                accuracy = quality_report['accuracy']
                
                # Logical consistency issues
                if accuracy['logical_consistency']:
                    st.markdown("#### Logical Consistency Issues")
                    for issue in accuracy['logical_consistency']:
                        st.warning(f"**{issue['issue']}**: {issue['count']} records ({issue['percentage']:.2f}%)")
                else:
                    st.success("No logical consistency issues found!")
                
                # High correlations
                if accuracy.get('high_correlations'):
                    st.markdown("#### 🔗 High Correlations (Potential Data Leakage)")
                    for corr in accuracy['high_correlations']:
                        st.info(f"**{corr['variables'][0]}** ↔ **{corr['variables'][1]}**: {corr['correlation']:.3f}")
          # Recommendations
        st.markdown('<h2 class="sub-header">📋 Data Quality Recommendations</h2>', unsafe_allow_html=True)
        
        if 'recommendations' in quality_report:
            # Summary overview
            total_recs = len(quality_report['recommendations'])
            high_count = len([r for r in quality_report['recommendations'] if r['priority'] == 'High'])
            medium_count = len([r for r in quality_report['recommendations'] if r['priority'] == 'Medium'])
            low_count = len([r for r in quality_report['recommendations'] if r['priority'] == 'Low'])
            
            st.markdown(f"""
            <div class="methodology-box">
                <h3>📊 Recommendations Summary</h3>
                <div style="display: flex; justify-content: space-around; margin: 1rem 0;">
                    <div style="text-align: center;">
                        <div style="color: #e50914; font-size: 2rem; font-weight: bold;">{high_count}</div>
                        <div style="color: #ffffff; font-size: 0.9rem;">High Priority</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #f9c23c; font-size: 2rem; font-weight: bold;">{medium_count}</div>
                        <div style="color: #ffffff; font-size: 0.9rem;">Medium Priority</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #46d369; font-size: 2rem; font-weight: bold;">{low_count}</div>
                        <div style="color: #ffffff; font-size: 0.9rem;">Low Priority</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #ffffff; font-size: 2rem; font-weight: bold;">{total_recs}</div>
                        <div style="color: #ffffff; font-size: 0.9rem;">Total Issues</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
              # Group recommendations by priority
            high_priority = [r for r in quality_report['recommendations'] if r['priority'] == 'High']
            medium_priority = [r for r in quality_report['recommendations'] if r['priority'] == 'Medium']
            low_priority = [r for r in quality_report['recommendations'] if r['priority'] == 'Low']
            
            # Debug: Show what priorities we have
            if st.sidebar.checkbox("Debug Mode - Show Recommendation Priorities"):
                st.sidebar.write("Available priorities:", [r['priority'] for r in quality_report['recommendations']])
                st.sidebar.write(f"High: {len(high_priority)}, Medium: {len(medium_priority)}, Low: {len(low_priority)}")
            
            # Display high priority recommendations first
            if high_priority:
                st.markdown("### 🔴 High Priority Issues")
                for rec in high_priority:
                    icon = "⚠️" if rec['category'] == 'Completeness' else "🔍" if rec['category'] == 'Validity' else "📊"
                    st.markdown(f"""
                    <div class="recommendation-card recommendation-high">
                        <div class="recommendation-header">
                            <div>
                                <span class="priority-badge">High Priority</span>
                                <span class="category-tag">{rec['category']}</span>
                            </div>
                            <span class="recommendation-icon">{icon}</span>
                        </div>
                        <div class="recommendation-content">
                            <div class="recommendation-issue">
                                <strong>🚨 Issue:</strong> {rec['issue']}
                            </div>
                            <div class="recommendation-solution">
                                <strong>💡 Recommendation:</strong> {rec['recommendation']}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
              # Display medium priority recommendations
            if medium_priority:
                st.markdown("### 🟡 Medium Priority Issues")
                for rec in medium_priority:
                    icon = "⚠️" if rec['category'] == 'Completeness' else "🔍" if rec['category'] == 'Validity' else "📊"
                    st.markdown(f"""
                    <div class="recommendation-card recommendation-medium">
                        <div class="recommendation-header">
                            <div>
                                <span class="priority-badge">Medium Priority</span>
                                <span class="category-tag">{rec['category']}</span>
                            </div>
                            <span class="recommendation-icon">{icon}</span>
                        </div>
                        <div class="recommendation-content">
                            <div class="recommendation-issue">
                                <strong>⚠️ Issue:</strong> {rec['issue']}
                            </div>
                            <div class="recommendation-solution">
                                <strong>💡 Recommendation:</strong> {rec['recommendation']}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            elif high_priority or low_priority:  # Only show this message if there are other priorities but no medium
                st.markdown("### 🟡 Medium Priority Issues")
                st.info("No medium priority issues found. Great job on data quality! 👍")
              # Display low priority recommendations
            if low_priority:
                st.markdown("### 🟢 Low Priority Issues")
                for rec in low_priority:
                    icon = "⚠️" if rec['category'] == 'Completeness' else "🔍" if rec['category'] == 'Validity' else "📊"
                    st.markdown(f"""
                    <div class="recommendation-card recommendation-low">
                        <div class="recommendation-header">
                            <div>
                                <span class="priority-badge">Low Priority</span>
                                <span class="category-tag">{rec['category']}</span>
                            </div>
                            <span class="recommendation-icon">{icon}</span>
                        </div>
                        <div class="recommendation-content">
                            <div class="recommendation-issue">
                                <strong>ℹ️ Issue:</strong> {rec['issue']}
                            </div>
                            <div class="recommendation-solution">
                                <strong>💡 Recommendation:</strong> {rec['recommendation']}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            elif high_priority or medium_priority:  # Only show this message if there are other priorities but no low
                st.markdown("### 🟢 Low Priority Issues")
                st.success("No low priority issues found")
        else:
            st.info("No specific recommendations available. Run data quality assessment to generate recommendations.")
        
        # Download quality report
        quality_report_json = pd.Series(quality_report).to_json()
        st.download_button(
            label="Download Quality Report",
            data=quality_report_json,
            file_name="data_quality_report.json",
            mime="application/json"
        )

def create_feature_analysis_page():
    """Create detailed feature analysis page"""
    st.markdown('<h1 class="main-header">Feature Analysis</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Data Overview page.")
        return
    
    df = st.session_state.original_data
    visualizer = HeartDiseaseVisualizer()
    
    # Feature selection
    st.sidebar.markdown("### Feature Selection")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Feature Engineering"])
    
    with tab1:
        st.markdown('<h2 class="sub-header"> Univariate Feature Analysis</h2>', unsafe_allow_html=True)
        
        # Select feature for detailed analysis
        selected_feature = st.selectbox("Select feature for detailed analysis:", options=df.columns.tolist())
        
        if selected_feature:
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature statistics
                st.markdown(f"#### {selected_feature} Statistics")
                
                if df[selected_feature].dtype in ['int64', 'float64']:
                    stats_df = df[selected_feature].describe().round(2)
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Distribution plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(data=df, x=selected_feature, kde=True, ax=ax)
                    ax.set_title(f'Distribution of {selected_feature}')
                    st.pyplot(fig)
                    plt.close()
                
                else:
                    value_counts = df[selected_feature].value_counts()
                    st.dataframe(value_counts, use_container_width=True)
                    
                    # Bar plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    value_counts.plot(kind='bar', ax=ax)
                    ax.set_title(f'Distribution of {selected_feature}')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()
            
            with col2:
                # Feature vs target analysis
                if 'num' in df.columns:
                    st.markdown(f"#### {selected_feature} vs Heart Disease")
                    
                    if df[selected_feature].dtype in ['int64', 'float64']:
                        # Box plot for numeric features
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.boxplot(data=df, x='num', y=selected_feature, ax=ax)
                        ax.set_title(f'{selected_feature} by Heart Disease Status')
                        st.pyplot(fig)
                        plt.close()
                    else:
                        # Cross-tabulation for categorical features
                        crosstab = pd.crosstab(df[selected_feature], df['num'])
                        fig, ax = plt.subplots(figsize=(8, 6))
                        crosstab.plot(kind='bar', ax=ax)
                        ax.set_title(f'{selected_feature} vs Heart Disease')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close()
                        
                        # Show crosstab table
                        st.dataframe(crosstab, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Bivariate Feature Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("Select X-axis feature:", options=numeric_cols, key="x_feature")
        
        with col2:
            y_feature = st.selectbox("Select Y-axis feature:", options=numeric_cols, key="y_feature")
        
        hue_feature = st.selectbox("Select color feature (optional):", options=['None'] + df.columns.tolist())
        
        if x_feature != y_feature:
            hue_col = None if hue_feature == 'None' else hue_feature
            bivariate_fig = visualizer.create_bivariate_analysis(df, x_feature, y_feature, hue_col)
            st.plotly_chart(bivariate_fig, use_container_width=True)
            
            # Correlation between selected features
            correlation = df[[x_feature, y_feature]].corr().iloc[0, 1]
            st.metric("Correlation", f"{correlation:.3f}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Feature Engineering</h2>', unsafe_allow_html=True)
        
        if st.button("Create Derived Features"):
            with st.spinner("Creating derived features..."):
                preprocessor = HeartDiseasePreprocessor()
                df_with_features, derived_features = preprocessor.create_derived_features(df.copy())
                
                st.success("Derived features created successfully!")
                
                # Show new features
                st.markdown("#### New Features Created")
                for feature, description in derived_features.items():
                    st.write(f"**{feature}**: {description}")
                
                # Preview new features
                new_cols = list(derived_features.keys())
                if new_cols:
                    st.markdown("#### Preview of New Features")
                    st.dataframe(df_with_features[new_cols].head(10), use_container_width=True)
                    
                    # Analysis of new features
                    for feature in new_cols:
                        if df_with_features[feature].dtype == 'category':
                            fig, ax = plt.subplots(figsize=(10, 6))
                            feature_counts = df_with_features[feature].value_counts()
                            feature_counts.plot(kind='bar', ax=ax)
                            ax.set_title(f'Distribution of {feature}')
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                            plt.close()

def create_model_training_page():
    """Create model training and evaluation page"""
    st.markdown('<h1 class="main-header">Model Training & Evaluation</h1>', unsafe_allow_html=True)
    
    # Clear sidebar content for this page
    st.sidebar.empty()
    
    if not st.session_state.get('preprocessing_complete', False):
        st.warning("Please complete data preprocessing first.")
        return
    
    df = st.session_state['processed_data']
    if 'num' not in df.columns:
        st.error("Target variable 'num' not found in processed data.")
        return
    
    # Feature/target split
    X = df.drop('num', axis=1)
    y = df['num']
    
    # Train/test split
    test_size = st.slider("Test set size (%)", 10, 50, 20, step=5) / 100
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    st.markdown("### 1. Model Selection & Training")
    st.info("Three models will be trained: SVM, XGBoost, and Random Forest.")
    
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models..."):
            # SVM
            svm = SVC(probability=True, random_state=random_state)
            svm.fit(X_train, y_train)
            svm_pred = svm.predict(X_test)
            svm_acc = accuracy_score(y_test, svm_pred)
            svm_report = classification_report(y_test, svm_pred, output_dict=True)
            
            # XGBoost
            xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)
            xgb_clf.fit(X_train, y_train)
            xgb_pred = xgb_clf.predict(X_test)
            xgb_acc = accuracy_score(y_test, xgb_pred)
            xgb_report = classification_report(y_test, xgb_pred, output_dict=True)
            
            # Random Forest
            rf = RandomForestClassifier(random_state=random_state)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_acc = accuracy_score(y_test, rf_pred)
            rf_report = classification_report(y_test, rf_pred, output_dict=True)
            
            # Store results
            results = {
                'SVM': {'acc': svm_acc, 'report': svm_report, 'pred': svm_pred},
                'XGBoost': {'acc': xgb_acc, 'report': xgb_report, 'pred': xgb_pred},
                'Random Forest': {'acc': rf_acc, 'report': rf_report, 'pred': rf_pred}
            }
            best_model = max(results, key=lambda k: results[k]['acc'])
            st.session_state['model_results'] = results
            st.session_state['best_model'] = best_model
            st.success(f"Best model: {best_model} (Accuracy: {results[best_model]['acc']:.3f})")
            
            # Show summary table
            summary_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[m]['acc'] for m in results]
            })
            st.dataframe(summary_df, use_container_width=True)
            
            # Show classification report for best model
            st.markdown(f"### 2. {best_model} - Detailed Results")
            st.text(classification_report(y_test, results[best_model]['pred']))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, results[best_model]['pred'])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{best_model} Confusion Matrix')
            st.pyplot(fig)
            plt.close()
            
            # Analysis
            st.markdown("### 3. Model Analysis & Insights")
            st.write(f"The best model is **{best_model}** with an accuracy of **{results[best_model]['acc']:.2%}**. ")
            st.write("\n**Key Insights:**")
            st.write("- The confusion matrix shows the distribution of correct and incorrect predictions across all classes.")
            st.write("- Review precision, recall, and F1-score for each class in the detailed report above.")
            st.write("- Consider feature importance (for tree-based models) for further analysis.")
            if best_model in ['XGBoost', 'Random Forest']:
                importances = (xgb_clf.feature_importances_ if best_model == 'XGBoost' else rf.feature_importances_)
                feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
                st.markdown("#### Top 5 Important Features")
                st.dataframe(feat_imp.head(5).to_frame('Importance'))

def create_portfolio_showcase_page():
    """Create a comprehensive portfolio showcase page"""
    st.markdown('<h1 class="main-header">Data Mining Portfolio Showcase</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Data Overview page.")
        return
    
    df = st.session_state.original_data
    visualizer = HeartDiseaseVisualizer()
    
    # Portfolio navigation
    portfolio_tabs = st.tabs([
        "Interactive Analytics", 
        "Machine Learning", 
        "Advanced Visualizations",
        "Real-time Filtering",
        "Business Insights"
    ])
    
    with portfolio_tabs[0]:
        st.markdown("### Interactive Data Analytics Dashboard")
        st.markdown("*Demonstrating real-time data interaction capabilities*")
          # Dynamic filtering section
        st.markdown("####Dynamic Filters")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            age_filter = st.slider("Age Range",                                 int(df['age'].min()), 
                                 int(df['age'].max()), 
                                 (int(df['age'].min()), int(df['age'].max())),
                                 key="portfolio_age_filter")
        
        with filter_col2:
            if 'sex' in df.columns:
                sex_filter = st.multiselect("Gender", 
                                          df['sex'].unique(), 
                                          default=df['sex'].unique(),
                                          key="portfolio_sex_filter")
            else:
                sex_filter = []
        
        with filter_col3:
            if 'cp' in df.columns:
                cp_filter = st.multiselect("Chest Pain Type", 
                                         df['cp'].unique(), 
                                         default=df['cp'].unique(),
                                         key="portfolio_cp_filter")
            else:
                cp_filter = []
        
        # Apply filters
        filtered_df = df[
            (df['age'] >= age_filter[0]) & 
            (df['age'] <= age_filter[1])
        ]
        
        if sex_filter and 'sex' in df.columns:
            filtered_df = filtered_df[filtered_df['sex'].isin(sex_filter)]
        
        if cp_filter and 'cp' in df.columns:
            filtered_df = filtered_df[filtered_df['cp'].isin(cp_filter)]
          # Show filter results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", len(filtered_df))
        with col2:
            disease_rate = (filtered_df['num'] > 0).mean() * 100 if 'num' in filtered_df.columns else 0
            st.metric("Disease Rate", f"{disease_rate:.1f}%")
        with col3:
            avg_age = filtered_df['age'].mean() if 'age' in filtered_df.columns else 0
            st.metric("Average Age", f"{avg_age:.1f}")
        with col4:
            avg_chol = filtered_df['chol'].mean() if 'chol' in filtered_df.columns else 0
            st.metric("Avg Cholesterol", f"{avg_chol:.0f}")
        
        # Interactive visualizations
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # 3D Scatter plot
            if all(col in filtered_df.columns for col in ['age', 'chol', 'thalch', 'num']):
                # Clean data for plotting to handle NaN values
                plot_df = clean_dataframe_for_plotting(filtered_df, ['age', 'chol', 'thalch', 'trestbps'])
                
                # Set size column
                size_col = 'trestbps' if 'trestbps' in plot_df.columns else None
                
                fig_3d = px.scatter_3d(
                    plot_df,
                    x='age',
                    y='chol', 
                    z='thalch',
                    color='num',
                    size=size_col,
                    hover_data=['oldpeak'] if 'oldpeak' in plot_df.columns else None,
                    title="3D Patient Analysis",
                    labels={'age': 'Age (years)', 'chol': 'Cholesterol', 'thalch': 'Max Heart Rate'},
                    color_continuous_scale="Viridis"
                )
                fig_3d.update_layout(height=500)
                st.plotly_chart(fig_3d, use_container_width=True)
        
        with viz_col2:
            # Dynamic correlation network
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns[:8]  # Limit for performance
            corr_matrix = filtered_df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Real-time Correlation Matrix",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with portfolio_tabs[1]:
        st.markdown("### Machine Learning Showcase")
        st.markdown("*Advanced ML algorithms with interactive predictions*")
        
        if 'num' not in df.columns:
            st.error("Target variable 'num' not found")
            return
        
        # ML Model comparison
        ml_col1, ml_col2 = st.columns(2)
        
        with ml_col1:
            st.markdown("#### Model Performance Comparison")

            # Prepare data
            X = df.select_dtypes(include=[np.number]).drop('num', axis=1)
            y = df['num']
            X = X.fillna(X.mean())
            
            # Quick model comparison
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
                'XGBoost': None  # Will handle import error gracefully
            }
            
            model_results = {}
            for name, model in models.items():
                if model is not None:
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        model_results[name] = accuracy
                    except:
                        continue
            
            if model_results:
                results_df = pd.DataFrame(list(model_results.items()), columns=['Model', 'Accuracy'])
                fig_models = px.bar(
                    results_df,
                    x='Model',
                    y='Accuracy',
                    title="ML Model Accuracy Comparison",
                    color='Accuracy',
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_models, use_container_width=True)
        
        with ml_col2:
            st.markdown("#### Interactive Prediction Tool")
            
            # Feature input for prediction
            if len(X.columns) >= 4:
                input_col1, input_col2 = st.columns(2)
                
                user_input = {}
                feature_cols = X.columns[:6]  # Use first 6 features
                
                for i, col in enumerate(feature_cols):
                    with input_col1 if i % 2 == 0 else input_col2:
                        min_val = float(X[col].min())
                        max_val = float(X[col].max())
                        mean_val = float(X[col].mean())
                        user_input[col] = st.number_input(
                            f"{col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            key=f"ml_pred_{col}"
                        )
                
                if st.button("Predict Heart Disease Risk", key="ml_predict"):
                    if model_results:
                        # Use the best model
                        best_model_name = max(model_results, key=model_results.get)
                        if best_model_name == 'Random Forest':
                            model = RandomForestClassifier(n_estimators=50, random_state=42)
                            model.fit(X_train, y_train)
                            
                            # Create prediction input
                            pred_input = pd.DataFrame([user_input])
                            
                            # Add missing columns with mean values
                            for col in X.columns:
                                if col not in pred_input.columns:
                                    pred_input[col] = X[col].mean()
                            
                            pred_input = pred_input[X.columns]
                            
                            # Make prediction
                            prediction = model.predict(pred_input)[0]
                            probabilities = model.predict_proba(pred_input)[0]
                            
                            if prediction == 0:
                                st.success(f"Low Risk: {probabilities[0]:.1%} confidence")
                            else:
                                st.error(f"High Risk: {probabilities[1] if len(probabilities) > 1 else probabilities[0]:.1%} confidence")

    with portfolio_tabs[2]:
        st.markdown("### Advanced Visualization Techniques")
        st.markdown("*Showcasing sophisticated data visualization skills*")
          # Advanced viz options
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Parallel Coordinates", "Sankey Diagram", "Sunburst Chart", "3D Surface Plot"],
            key="adv_viz_type"
        )
        
        if viz_type == "Parallel Coordinates":
            # Parallel coordinates plot
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
            if len(numeric_cols) > 0:
                # Clean data for parallel coordinates - only fill numeric columns
                df_clean = df.copy()
                numeric_means = df_clean[numeric_cols].mean()
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(numeric_means)
                
                fig_parallel = px.parallel_coordinates(
                    df_clean,
                    dimensions=numeric_cols,
                    color='num' if 'num' in df_clean.columns else numeric_cols[0],
                    title="Parallel Coordinates: Multi-dimensional Analysis"
                )
                st.plotly_chart(fig_parallel, use_container_width=True)
            else:
                st.warning("No numeric columns available for parallel coordinates plot.")
        
        elif viz_type == "Sankey Diagram":
            # Create Sankey diagram for categorical relationships
            if 'sex' in df.columns and 'cp' in df.columns and 'num' in df.columns:
                # Prepare data for Sankey
                sankey_data = df.groupby(['sex', 'cp', 'num']).size().reset_index(name='count')
                
                st.markdown("#### Patient Flow: Gender → Chest Pain → Heart Disease")
                st.write("This Sankey diagram shows the flow of patients through different categorical states.")
                
                # Simple representation since full Sankey requires more complex setup
                fig_flow = px.parallel_categories(
                    sankey_data,
                    dimensions=['sex', 'cp', 'num'],
                    color='count',
                    title="Patient Flow Analysis"
                )
                st.plotly_chart(fig_flow, use_container_width=True)
        
        elif viz_type == "Sunburst Chart":
            # Hierarchical data visualization            if all(col in df.columns for col in ['sex', 'cp', 'num']):
                # Create hierarchical grouping
                hierarchy_data = df.groupby(['sex', 'cp', 'num']).size().reset_index(name='count')
                
                fig_sunburst = px.sunburst(
                    hierarchy_data,
                    path=['sex', 'cp', 'num'],
                    values='count',
                )
                st.plotly_chart(fig_sunburst, use_container_width=True)
        
        # Advanced insights and recommendations
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            st.markdown("#### 📊 Advanced Analytics Insights")
            st.markdown("These sophisticated visualizations reveal:")
            st.markdown("- **Multi-dimensional patterns** in patient data")
            st.markdown("- **Flow dynamics** between categorical variables") 
            st.markdown("- **Hierarchical relationships** in risk factors")
            st.markdown("- **Complex correlations** across multiple features")

    with adv_col2:
        st.markdown("#### 🎯 Strategic Recommendations")

        recommendations = [
            {
                "icon": "🏥",
                "title": "Clinical Focus",
                "description": "Prioritize screening for patients over 50 with multiple risk factors"
            },
            {
                "icon": "🤖",
                "title": "Predictive Analytics",
                "description": "Implement ML models for early detection and risk assessment"
            },
            {
                "icon": "📊",
                "title": "Risk Stratification",
                "description": "Use cholesterol and blood pressure as key indicators for patient triage"
            },
            {
                    "icon": "📱",
                    "title": "Dashboard Implementation",
                    "description": "Deploy real-time monitoring systems for healthcare providers"
                },
                {
                    "icon": "🔄",
                    "title": "Continuous Learning",
                    "description": "Update models with new patient data for improved accuracy"
                }
            ]
            
        for rec in recommendations:
            st.markdown(f"""
            <div class="business-recommendation-card">
                <div class="business-rec-header">
                    <span class="recommendation-icon">{rec['icon']}</span>
                    {rec['title']}
                </div>
                <div class="business-rec-content">
                    {rec['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with portfolio_tabs[3]:
        st.markdown("### Real-time Data Filtering")
        st.markdown("*Demonstrating interactive data manipulation capabilities*")
        
        # Real-time filtering with multiple criteria
        filter_df = visualizer.create_interactive_filter_dashboard(df)
        
        # Show before/after comparison
        st.markdown("#### Before vs After Filtering")
        comparison_col1, comparison_col2 = st.columns(2)
        
        with comparison_col1:
            st.markdown("**Original Dataset**")
            st.metric("Records", len(df))
            st.metric("Disease Rate", f"{(df['num'] > 0).mean() * 100:.1f}%")
        
        with comparison_col2:
            st.markdown("**Filtered Dataset**") 
            st.metric("Records", len(filter_df))
            disease_rate_filtered = (filter_df['num'] > 0).mean() * 100 if 'num' in filter_df.columns else 0
            st.metric("Disease Rate", f"{disease_rate_filtered:.1f}%")
    
    with portfolio_tabs[4]:
        st.markdown("### Business Insights & Recommendations")
        st.markdown("*Translating data science findings into actionable business intelligence*")
        
        # Generate insights
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("#### Key Findings")
            
            # Calculate key insights
            total_patients = len(df)
            disease_rate = (df['num'] > 0).mean() * 100 if 'num' in df.columns else 0
            high_risk_age = df[df['num'] > 0]['age'].mean() if 'num' in df.columns and 'age' in df.columns else 0
            
            insights = [
                f"**Dataset Overview**: {total_patients:,} patient records analyzed",
                f"**Disease Prevalence**: {disease_rate:.1f}% of patients have heart disease",
                f"**High-Risk Demographics**: Average age of affected patients is {high_risk_age:.1f} years",
                f"**Data Quality**: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}% data completeness",
            ]
            
            for insight in insights:
                st.markdown(insight)
        
        with insights_col2:
            st.markdown("#### Recommendations")
            
            recommendations = [
                "**Clinical Focus**: Prioritize screening for patients over 50",
                "**Predictive Analytics**: Implement ML models for early detection", 
                "**Risk Stratification**: Use cholesterol and blood pressure as key indicators",
                "**Dashboard Implementation**: Deploy real-time monitoring systems",
                "**Continuous Learning**: Update models with new patient data"
            ]
            
            for rec in recommendations:
                st.markdown(rec)
        
        # ROI Calculation example
        st.markdown("#### Business Impact Estimation")
        
        impact_col1, impact_col2, impact_col3 = st.columns(3)
        
        with impact_col1:
            early_detection_rate = st.slider("Early Detection Rate (%)", 0, 100, 75)
        
        with impact_col2:
            cost_per_case = st.number_input("Cost per Missed Case ($)", value=50000)
        
        with impact_col3:
            implementation_cost = st.number_input("Implementation Cost ($)", value=100000)
        
        # Calculate ROI
        potential_cases = int(total_patients * disease_rate / 100)
        cases_prevented = int(potential_cases * early_detection_rate / 100)
        cost_savings = cases_prevented * cost_per_case
        roi = ((cost_savings - implementation_cost) / implementation_cost) * 100
        
        st.markdown("#### ROI Analysis")
        roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
        
        with roi_col1:
            st.metric("Cases Prevented", f"{cases_prevented:,}")
        with roi_col2:
            st.metric("Cost Savings", f"${cost_savings:,}")
        with roi_col3:
            st.metric("ROI", f"{roi:.1f}%")
        with roi_col4:
            payback_months = (implementation_cost / (cost_savings / 12)) if cost_savings > 0 else float('inf')
            st.metric("Payback Period", f"{payback_months:.1f} months" if payback_months != float('inf') else "N/A")

def create_automated_pipeline_page():
    """Create automated pipeline page for testing all combinations"""
    st.markdown('<h1 class="main-header">Automated ML Pipeline</h1>', unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Preprocessing Preview", "Run Pipeline", "Results Analysis"])
    
    with tab1:
        st.markdown("### Interactive Preprocessing Preview")
        st.markdown("Select preprocessing methods to see their effects on the data in real-time.")
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first by visiting the Data Overview tab.")
            return
        
        df = st.session_state.original_data.copy()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            null_method = st.selectbox(
                "Null Handling Method",
                ["intelligent", "drop_rows"],
                help="Choose how to handle missing values"
            )
        
        with col2:
            outlier_method = st.selectbox(
                "Outlier Handling Method", 
                ["cap", "remove"],
                help="Choose how to handle outliers"
            )
        
        with col3:
            encoding_method = st.selectbox(
                "Categorical Encoding",
                ["onehot", "label"],
                help="Choose encoding method for categorical variables"
            )
        
        # Apply preprocessing based on selections
        try:
            from automated_pipeline import AutomatedPipeline
            pipeline = AutomatedPipeline("data/heart_disease_uci.csv")
            pipeline.load_data()
            
            # Apply selected preprocessing
            df_processed = pipeline.original_data.copy()
            original_shape = df_processed.shape
            
            df_processed = pipeline.handle_nulls(df_processed, null_method)
            after_nulls_shape = df_processed.shape
            
            df_processed = pipeline.handle_outliers(df_processed, outlier_method)
            after_outliers_shape = df_processed.shape
            
            df_processed = pipeline.encode_categorical(df_processed, encoding_method)
            final_shape = df_processed.shape
            
            # Show preprocessing effects
            st.markdown("#### Preprocessing Impact")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Original Rows", 
                    original_shape[0],
                    help="Number of rows in original dataset"
                )
            
            with col2:
                rows_lost_nulls = original_shape[0] - after_nulls_shape[0]
                st.metric(
                    "After Null Handling", 
                    after_nulls_shape[0],
                    delta=-rows_lost_nulls if rows_lost_nulls > 0 else None,
                    help="Rows remaining after null handling"
                )
            
            with col3:
                rows_lost_outliers = after_nulls_shape[0] - after_outliers_shape[0]
                st.metric(
                    "After Outlier Handling", 
                    after_outliers_shape[0],
                    delta=-rows_lost_outliers if rows_lost_outliers > 0 else None,
                    help="Rows remaining after outlier handling"
                )
            
            with col4:
                features_change = final_shape[1] - original_shape[1]
                st.metric(
                    "Final Features", 
                    final_shape[1],
                    delta=features_change if features_change != 0 else None,
                    help="Number of features after encoding"
                )
            
            # Interactive data distribution visualization
            st.markdown("#### Data Distribution Analysis")
            
            # Feature selection for visualization
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            if 'num' in numeric_cols:
                numeric_cols.remove('num')
            
            if numeric_cols:
                selected_feature = st.selectbox(
                    "Select feature to visualize:",
                    numeric_cols,
                    help="Choose a feature to see its distribution"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Original distribution
                    if selected_feature in st.session_state.original_data.columns:
                        fig_orig = px.histogram(
                            st.session_state.original_data,
                            x=selected_feature,
                            title=f"Original {selected_feature} Distribution",
                            nbins=30
                        )
                        fig_orig.update_layout(showlegend=False)
                        st.plotly_chart(fig_orig, use_container_width=True)
                
                with col2:
                    # Processed distribution
                    fig_proc = px.histogram(
                        df_processed,
                        x=selected_feature,
                        title=f"Processed {selected_feature} Distribution",
                        nbins=30,
                        color_discrete_sequence=['#ff7f0e']
                    )
                    fig_proc.update_layout(showlegend=False)
                    st.plotly_chart(fig_proc, use_container_width=True)
                
                # Box plot comparison
                st.markdown("##### Box Plot Comparison")
                
                # Create comparison dataframe
                orig_data = st.session_state.original_data[selected_feature].dropna()
                proc_data = df_processed[selected_feature].dropna()
                
                comparison_df = pd.DataFrame({
                    'Value': list(orig_data) + list(proc_data),
                    'Type': ['Original'] * len(orig_data) + ['Processed'] * len(proc_data)
                })
                
                fig_box = px.box(
                    comparison_df,
                    x='Type',
                    y='Value',
                    title=f"{selected_feature} - Before vs After Processing"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Target distribution
            st.markdown("#### Target Variable Distribution")
            
            if 'num' in df_processed.columns:
                # Convert to binary for visualization
                y_binary = (df_processed['num'] > 0).astype(int)
                target_counts = y_binary.value_counts()
                
                fig_target = px.pie(
                    values=target_counts.values,
                    names=['No Disease', 'Disease'],
                    title="Target Distribution After Preprocessing"
                )
                st.plotly_chart(fig_target, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("No Disease", target_counts[0])
                with col2:
                    st.metric("Disease", target_counts[1])
            
        except Exception as e:
            st.error(f"Error in preprocessing preview: {str(e)}")
    
    with tab2:
        st.markdown("### Run Automated Pipeline")
        
        st.markdown("""
        <div class="methodology-box">
            <h3>Automated Pipeline Overview</h3>
            <p>This pipeline automatically tests <strong>24 different combinations</strong> of preprocessing methods and machine learning models:</p>
            <ul>
                <li><strong>Null Handling:</strong> Intelligent Imputation, Drop Rows (2 methods)</li>
                <li><strong>Outlier Treatment:</strong> Capping, Removal (2 methods)</li>  
                <li><strong>Categorical Encoding:</strong> One-Hot Encoding, Label Encoding (2 methods)</li>
                <li><strong>ML Models:</strong> SVM, XGBoost, Random Forest (3 models)</li>
            </ul>
            <p><strong>Total Combinations:</strong> 2 × 2 × 2 × 3 = 24 experiments</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for pipeline results
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = None
        if 'pipeline_running' not in st.session_state:
            st.session_state.pipeline_running = False
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Start Automated Pipeline", type="primary", disabled=st.session_state.pipeline_running):
                st.session_state.pipeline_running = True
                
                # Create progress placeholders
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    with st.spinner("Initializing automated pipeline..."):
                        # Initialize pipeline
                        pipeline = AutomatedPipeline("data/heart_disease_uci.csv")
                        
                        status_text.text("Loading data...")
                        if not pipeline.load_data():
                            st.error("Failed to load data!")
                            st.session_state.pipeline_running = False
                            return
                        
                        status_text.text("Starting automated testing...")
                        
                        # Run pipeline with progress updates
                        import itertools
                        combinations = list(itertools.product(
                            pipeline.null_methods,
                            pipeline.outlier_methods, 
                            pipeline.encoding_methods,
                            pipeline.models
                        ))
                        
                        total_combinations = len(combinations)
                        completed = 0
                        results = []
                        skipped_combinations = []
                        
                        for i, (null_method, outlier_method, encoding_method, model_name) in enumerate(combinations, 1):
                            try:
                                status_text.text(f"Testing combination {i}/{total_combinations}: "
                                                f"Nulls={null_method}, Outliers={outlier_method}, "
                                                f"Encoding={encoding_method}, Model={model_name}")
                                
                                # Start with fresh data
                                df_processed = pipeline.original_data.copy()
                                
                                # Apply preprocessing steps
                                df_processed = pipeline.handle_nulls(df_processed, null_method)
                                df_processed = pipeline.handle_outliers(df_processed, outlier_method)
                                df_processed = pipeline.encode_categorical(df_processed, encoding_method)
                                
                                # Prepare features and target
                                X, y = pipeline.prepare_features_target(df_processed)
                                
                                # Lower threshold to avoid skipping combinations
                                if len(X) < 30:
                                    skipped_combinations.append({
                                        'combination_id': i,
                                        'reason': f'Insufficient samples: {len(X)}',
                                        'null_handling': null_method,
                                        'outlier_handling': outlier_method,
                                        'categorical_encoding': encoding_method,
                                        'model': model_name
                                    })
                                    completed += 1
                                    progress_bar.progress(completed / total_combinations)
                                    continue
                                
                                # Split data
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.2, random_state=42, stratify=y
                                )
                                
                                # Train model and get metrics
                                metrics = pipeline.train_model(X_train, X_test, y_train, y_test, model_name)
                                
                                # Store results
                                result = {
                                    'combination_id': i,
                                    'null_handling': null_method,
                                    'outlier_handling': outlier_method,
                                    'categorical_encoding': encoding_method,
                                    'model': model_name,
                                    'sample_size': len(X),
                                    'features_count': X.shape[1],
                                    'accuracy': metrics['accuracy'],
                                    'precision': metrics['precision'],
                                    'recall': metrics['recall'],
                                    'f1_score': metrics['f1_score'],
                                    'sensitivity': metrics['sensitivity']
                                }
                                
                                results.append(result)
                                completed += 1
                                progress_bar.progress(completed / total_combinations)
                                
                            except Exception as e:
                                skipped_combinations.append({
                                    'combination_id': i,
                                    'reason': f'Error: {str(e)}',
                                    'null_handling': null_method,
                                    'outlier_handling': outlier_method,
                                    'categorical_encoding': encoding_method,
                                    'model': model_name
                                })
                                completed += 1
                                progress_bar.progress(completed / total_combinations)
                                continue
                        
                        # Store results in session state
                        st.session_state.pipeline_results = results
                        st.session_state.skipped_combinations = skipped_combinations
                        
                        # Find best result
                        if results:
                            best_result = max(results, key=lambda x: x['f1_score'])
                            st.session_state.best_result = best_result
                            
                            # Save to JSON
                            output = {
                                'metadata': {
                                    'total_combinations_attempted': len(combinations),
                                    'successful_combinations': len(results),
                                    'skipped_combinations': len(skipped_combinations),
                                    'best_combination': best_result,
                                    'run_timestamp': pd.Timestamp.now().isoformat()
                                },
                                'results': results,
                                'skipped': skipped_combinations
                            }
                            
                            with open('pipeline_results.json', 'w') as f:
                                json.dump(output, f, indent=2)
                        
                        status_text.text("Completed successfully!")
                        progress_bar.progress(1.0)
                        
                        # Show summary
                        st.success(f"Completed! {len(results)} successful combinations out of {len(combinations)} total.")
                        if skipped_combinations:
                            st.warning(f"{len(skipped_combinations)} combinations were skipped due to insufficient data or errors.")
                        
                except Exception as e:
                    st.error(f"Pipeline failed: {str(e)}")
                
                finally:
                    st.session_state.pipeline_running = False
        
        with col2:
            st.markdown("### Pipeline Status")
            if st.session_state.pipeline_running:
                st.info("Pipeline is running...")
            elif st.session_state.pipeline_results:
                st.success(f"Completed! {len(st.session_state.pipeline_results)} combinations tested")
            else:
                st.info("Ready to run")
    
    with tab3:
        st.markdown("### Results Analysis")
        
        # Display results if available
        if st.session_state.pipeline_results:
            results_df = pd.DataFrame(st.session_state.pipeline_results)
            
            # Best result highlight
            if 'best_result' in st.session_state:
                best = st.session_state.best_result
                
                st.markdown("#### Best Performing Combination")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("F1 Score", f"{best['f1_score']:.4f}")
                with col2:
                    st.metric("Accuracy", f"{best['accuracy']:.4f}")
                with col3:
                    st.metric("Precision", f"{best['precision']:.4f}")
                with col4:
                    st.metric("Recall", f"{best['recall']:.4f}")
                
                st.markdown(f"""
                <div class="methodology-box">
                    <h4>Best Configuration</h4>
                    <ul>
                        <li><strong>Null Handling:</strong> {best['null_handling']}</li>
                        <li><strong>Outlier Treatment:</strong> {best['outlier_handling']}</li>
                        <li><strong>Categorical Encoding:</strong> {best['categorical_encoding']}</li>
                        <li><strong>Model:</strong> {best['model']}</li>
                        <li><strong>Sample Size:</strong> {best['sample_size']}</li>
                        <li><strong>Features:</strong> {best['features_count']}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance comparison charts
            st.markdown("#### Interactive Performance Analysis")

            tab1, tab2, tab3 = st.tabs(["Model Comparison", "Method Analysis", "Complete Results"])

            with tab1:
                # Model performance comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_f1 = px.box(results_df, x='model', y='f1_score', 
                                   title='F1 Score Distribution by Model',
                                   color='model')
                    fig_f1.update_layout(showlegend=False)
                    st.plotly_chart(fig_f1, use_container_width=True)
                
                with col2:
                    fig_acc = px.box(results_df, x='model', y='accuracy',
                                    title='Accuracy Distribution by Model',
                                    color='model')
                    fig_acc.update_layout(showlegend=False)
                    st.plotly_chart(fig_acc, use_container_width=True)

                # Scatter plot of accuracy vs f1_score
                # Clean data for plotting to handle NaN values
                scatter_df = clean_dataframe_for_plotting(results_df, ['sample_size'])
                
                fig_scatter = px.scatter(
                    scatter_df, 
                    x='accuracy', 
                    y='f1_score',
                    color='model',
                    size='sample_size',
                    hover_data=['null_handling', 'outlier_handling', 'categorical_encoding'],
                    title='Accuracy vs F1 Score by Model'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with tab2:
                # Method analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    # Null handling performance
                    null_perf = results_df.groupby('null_handling')['f1_score'].agg(['mean', 'std']).reset_index()
                    fig_null = px.bar(
                        null_perf, 
                        x='null_handling', 
                        y='mean',
                        error_y='std',
                        title='Average F1 Score by Null Handling Method',
                        color='null_handling'
                    )
                    fig_null.update_layout(showlegend=False)
                    st.plotly_chart(fig_null, use_container_width=True)
                    
                    # Outlier handling performance
                    outlier_perf = results_df.groupby('outlier_handling')['f1_score'].agg(['mean', 'std']).reset_index()
                    fig_outlier = px.bar(
                        outlier_perf, 
                        x='outlier_handling', 
                        y='mean',
                        error_y='std',
                        title='Average F1 Score by Outlier Handling Method',
                        color='outlier_handling'
                    )
                    fig_outlier.update_layout(showlegend=False)
                    st.plotly_chart(fig_outlier, use_container_width=True)
                
                with col2:
                    # Encoding performance
                    encoding_perf = results_df.groupby('categorical_encoding')['f1_score'].agg(['mean', 'std']).reset_index()
                    fig_encoding = px.bar(
                        encoding_perf, 
                        x='categorical_encoding', 
                        y='mean',
                        error_y='std',
                        title='Average F1 Score by Encoding Method',
                        color='categorical_encoding'
                    )
                    fig_encoding.update_layout(showlegend=False)
                    st.plotly_chart(fig_encoding, use_container_width=True)
                    
                    # Heatmap of method combinations
                    pivot_table = results_df.pivot_table(
                        values='f1_score', 
                        index='null_handling', 
                        columns='outlier_handling',
                        aggfunc='mean'
                    )
                    fig_heatmap = px.imshow(
                        pivot_table, 
                        title='F1 Score Heatmap: Null vs Outlier Methods',
                        aspect='auto',
                        color_continuous_scale='RdYlBu_r'
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with tab3:
                # Complete results table
                st.markdown("#### All Combinations Results")
                
                # Sort by F1 score
                display_df = results_df.sort_values('f1_score', ascending=False)
                
                # Format for display
                display_df_formatted = display_df.copy()
                for col in ['accuracy', 'precision', 'recall', 'f1_score', 'sensitivity']:
                    display_df_formatted[col] = display_df_formatted[col].round(4)
                
                st.dataframe(
                    display_df_formatted,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button for results
                json_str = json.dumps(st.session_state.pipeline_results, indent=2)
                st.download_button(
                    label="Download Results as JSON",
                    data=json_str,
                    file_name=f"pipeline_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                # Show skipped combinations if any
                if 'skipped_combinations' in st.session_state and st.session_state.skipped_combinations:
                    st.markdown("#### Skipped Combinations")
                    skipped_df = pd.DataFrame(st.session_state.skipped_combinations)
                    st.dataframe(skipped_df, use_container_width=True, hide_index=True)
        
        else:
            st.info("Run the automated pipeline first to see results here!")
            st.markdown("""
            <div class="methodology-box">
                <h4>What you'll see here after running the pipeline:</h4>
                <ul>
                    <li><strong>Best Performing Model:</strong> Highlights of the top combination</li>
                    <li><strong>Interactive Charts:</strong> Compare models and preprocessing methods</li>
                    <li><strong>Performance Metrics:</strong> Accuracy, Precision, Recall, F1-Score, Sensitivity</li>
                    <li><strong>Detailed Results:</strong> Complete table of all 24 combinations</li>                    <li><strong>Download Options:</strong> Export results as JSON for further analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Top-level tab navigation for better UX
    pages = {        "Data Overview": create_data_overview_page,
        "Exploratory Analysis": create_exploratory_analysis_page,
        "Data Quality": create_quality_assessment_page,
        "Automated Pipeline": create_automated_pipeline_page,
        "Portfolio Showcase": create_portfolio_showcase_page
    }
    
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        margin-bottom: 2rem;
        background: #000000;
        border-radius: 4px;
        padding: 0.5rem;
        border: 1px solid #333333;
    }
    </style>
    """, unsafe_allow_html=True)
    
    tab_labels = list(pages.keys())
    tabs = st.tabs(tab_labels)
    for i, tab in enumerate(tabs):
        with tab:
            pages[tab_labels[i]]()    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #ffffff; padding: 2rem; background: #141414; border-radius: 4px; margin-top: 2rem; border: 1px solid #333333;">
        <p style="font-size: 1.2rem; font-weight: 700; color: #e50914; font-family: 'Helvetica Neue', Arial, sans-serif;">🫀 Heart Disease Data Mining Platform</p>
        <p style="color: #ffffff; font-family: 'Helvetica Neue', Arial, sans-serif;">Comprehensive data preprocessing, analysis, and modeling for healthcare analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

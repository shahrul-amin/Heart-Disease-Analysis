"""
Heart Disease Data Preprocessing Pipeline
Comprehensive data preprocessing with methodology documentation
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePreprocessor:
    """
    Comprehensive preprocessing pipeline for heart disease dataset
    Implements best practices for data mining tasks
    """
    
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.preprocessing_log = []
        self.column_mappings = {}
        self.imputation_values = {}
        self.outlier_info = {}
        
    def load_data(self, file_path):
        """Load and perform initial data assessment"""
        try:
            self.original_data = pd.read_csv(file_path, index_col='id')
            self.log_step("Data loaded successfully", f"Shape: {self.original_data.shape}")
            return True
        except Exception as e:
            self.log_step("Data loading failed", str(e))
            return False
    
    def log_step(self, step, details=""):
        """Log preprocessing steps for methodology documentation"""
        self.preprocessing_log.append({
            'step': step,
            'details': details,
            'timestamp': pd.Timestamp.now()
        })
    
    def get_data_quality_report(self):
        """Generate comprehensive data quality assessment"""
        if self.original_data is None:
            return None
            
        report = {
            'basic_info': {
                'rows': self.original_data.shape[0],
                'columns': self.original_data.shape[1],
                'memory_usage': self.original_data.memory_usage(deep=True).sum()
            },
            'missing_values': {
                'count': self.original_data.isnull().sum().to_dict(),
                'percentage': (self.original_data.isnull().sum() / len(self.original_data) * 100).to_dict()
            },
            'data_types': self.original_data.dtypes.to_dict(),
            'duplicates': self.original_data.duplicated().sum(),
            'unique_values': {col: self.original_data[col].nunique() for col in self.original_data.columns}
        }
        
        # Add statistical summary for numeric columns
        numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report['statistics'] = self.original_data[numeric_cols].describe().to_dict()
        
        return report
    
    def identify_column_types(self):
        """Identify and categorize columns by type and purpose"""
        df = self.original_data.copy()
        
        # Define column categories based on domain knowledge
        column_types = {
            'target': ['num'],
            'demographic': ['age', 'sex'],
            'clinical_continuous': ['trestbps', 'chol', 'thalch', 'oldpeak'],
            'clinical_categorical': ['cp', 'restecg', 'slope', 'thal'],
            'clinical_binary': ['fbs', 'exang'],
            'diagnostic': ['ca'],
            'metadata': ['dataset']
        }
        
        # Automatically detect data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        boolean_cols = []
        
        # Identify boolean columns stored as text
        for col in categorical_cols:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({'TRUE', 'FALSE', 'True', 'False', '1', '0'}):
                boolean_cols.append(col)
        
        auto_detection = {
            'numeric': numeric_cols,
            'categorical': [col for col in categorical_cols if col not in boolean_cols],
            'boolean': boolean_cols
        }
        
        self.column_mappings = {
            'types': column_types,
            'auto_detected': auto_detection
        }
        
        self.log_step("Column types identified", f"Categories: {list(column_types.keys())}")
        return column_types, auto_detection
    
    def standardize_data_types(self, df_input=None):
        """Standardize data types based on domain knowledge"""
        if df_input is None:
            df_input = self.original_data
            
        if df_input is None:
            return None
            
        df = df_input.copy()
        type_changes = {}
        
        # Convert boolean columns (handle missing values first)
        boolean_mappings = {
            'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0,
            'YES': 1, 'NO': 0, 'Yes': 1, 'No': 0
        }
        
        for col in ['fbs', 'exang']:
            if col in df.columns:
                original_type = str(df[col].dtype)
                # Only convert non-null values
                mask = df[col].notna()
                df.loc[mask, col] = df.loc[mask, col].map(boolean_mappings)
                # Convert to numeric, keeping NaN as NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                type_changes[col] = f"{original_type} -> numeric"
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'num']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize categorical columns
        categorical_cols = ['sex', 'cp', 'restecg', 'slope', 'thal', 'dataset']
        for col in categorical_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype('category')
                type_changes[col] = f"object -> category"
        
        self.log_step("Data types standardized", f"Changes: {len(type_changes)} columns")
        return df, type_changes
    
    def handle_missing_values(self, df_input=None, strategy='intelligent'):
        """
        Handle missing values using data-driven approach
        Strategy options: 'intelligent', 'drop', 'simple'
        """
        if df_input is None:
            df_input = self.original_data
            
        if df_input is None:
            return None
            
        df = df_input.copy()
        missing_report = {}
        
        # Get data distributions to determine best imputation strategy
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                
                if strategy == 'intelligent':
                    if col in numeric_cols:
                        # Check skewness to determine mean vs median
                        skewness = df[col].skew()
                        if abs(skewness) > 0.5:
                            # Use median for skewed data
                            fill_value = df[col].median()
                            method = 'median'
                        else:
                            # Use mean for symmetric data
                            fill_value = df[col].mean()
                            method = 'mean'
                    else:
                        # Use mode for categorical data
                        fill_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                        method = 'mode'
                    
                    df[col].fillna(fill_value, inplace=True)
                    self.imputation_values[col] = {'value': fill_value, 'method': method}
                    
                elif strategy == 'drop':
                    df.dropna(subset=[col], inplace=True)
                    method = 'dropped'
                
                missing_report[col] = {
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct,
                    'method': method
                }
        
        self.log_step("Missing values handled", f"Strategy: {strategy}, Columns affected: {len(missing_report)}")
        return df, missing_report
    
    def detect_outliers(self, df_input=None, method='iqr', threshold=1.5):
        """
        Detect outliers using specified method
        Methods: 'iqr', 'zscore', 'isolation_forest'
        """
        if df_input is None:
            df_input = self.original_data
            
        if df_input is None:
            return None
            
        df = df_input.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            if col != 'num':  # Skip target variable
                outliers_idx = []
                
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outliers_idx = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outliers_idx = df[col].dropna().iloc[np.where(z_scores > threshold)[0]].index.tolist()
                
                outlier_info[col] = {
                    'count': len(outliers_idx),
                    'percentage': (len(outliers_idx) / len(df)) * 100,
                    'indices': outliers_idx
                }
        
        self.outlier_info = outlier_info
        self.log_step("Outliers detected", f"Method: {method}, Total outliers found: {sum([info['count'] for info in outlier_info.values()])}")
        return outlier_info
    
    def handle_outliers(self, df_input=None, method='cap', outlier_info=None):
        """
        Handle outliers using specified method
        Methods: 'cap', 'remove', 'transform'
        """
        if df_input is None:
            df_input = self.original_data
            
        if outlier_info is None:
            outlier_info = self.outlier_info
            
        df = df_input.copy()
        treatment_report = {}
        
        for col, info in outlier_info.items():
            if info['count'] > 0:
                if method == 'cap':
                    # Cap outliers to reasonable bounds
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    treatment_report[col] = f"Capped to [{lower_bound:.2f}, {upper_bound:.2f}]"
                    
                elif method == 'remove':
                    # Remove outlier rows
                    df = df.drop(info['indices'])
                    treatment_report[col] = f"Removed {info['count']} outlier rows"
                    
                elif method == 'transform':
                    # Log transform for positive skewed data
                    if df[col].min() > 0:
                        df[col] = np.log1p(df[col])
                        treatment_report[col] = "Applied log transformation"
        
        self.log_step("Outliers handled", f"Method: {method}, Columns treated: {len(treatment_report)}")
        return df, treatment_report
    
    def create_derived_features(self, df):
        """Create meaningful derived features"""
        derived_features = {}
        
        # Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 35, 45, 55, 65, 100], 
                                   labels=['Young', 'Middle-aged', 'Mature', 'Senior', 'Elderly'])
            derived_features['age_group'] = "Categorical age groups"
        
        # Cholesterol categories
        if 'chol' in df.columns:
            df['chol_category'] = pd.cut(df['chol'], 
                                       bins=[0, 200, 240, float('inf')], 
                                       labels=['Normal', 'Borderline', 'High'])
            derived_features['chol_category'] = "Cholesterol risk categories"
        
        # Blood pressure categories
        if 'trestbps' in df.columns:
            df['bp_category'] = pd.cut(df['trestbps'], 
                                     bins=[0, 120, 140, 180, float('inf')], 
                                     labels=['Normal', 'Elevated', 'High', 'Crisis'])
            derived_features['bp_category'] = "Blood pressure categories"
        
        # Binary target for classification
        if 'num' in df.columns:
            df['heart_disease'] = (df['num'] > 0).astype(int)
            derived_features['heart_disease'] = "Binary heart disease indicator"
        
        self.log_step("Derived features created", f"New features: {list(derived_features.keys())}")
        return df, derived_features
    
    def encode_categorical_variables(self, df, encoding_method='onehot'):
        """Encode categorical variables for machine learning"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        encoding_info = {}
        
        if encoding_method == 'onehot':
            # One-hot encoding for categorical variables
            df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dtype=int)
            encoding_info = {col: 'one-hot encoded' for col in categorical_cols}
            
        elif encoding_method == 'label':
            # Label encoding
            df_encoded = df.copy()
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
                encoding_info[col] = f'label encoded ({len(le.classes_)} classes)'
        
        self.log_step("Categorical encoding completed", f"Method: {encoding_method}, Columns: {len(categorical_cols)}")
        return df_encoded, encoding_info
    
    def run_full_pipeline(self, file_path, missing_strategy='intelligent', 
                         outlier_method='cap', encoding_method='onehot'):
        """
        Run the complete preprocessing pipeline
        """
        self.preprocessing_log = []  # Reset log
        
        # Step 1: Load data
        if not self.load_data(file_path):
            return None
        
        # Step 2: Identify column types
        self.identify_column_types()
        
        # Step 3: Initial data type standardization (without converting boolean to int yet)
        df, type_changes = self.standardize_data_types()
        
        # Step 4: Handle missing values FIRST
        df, missing_report = self.handle_missing_values(df, strategy=missing_strategy)
        
        # Step 5: Now convert boolean columns to integers (after missing values are handled)
        for col in ['fbs', 'exang']:
            if col in df.columns:
                # Convert to int now that NaN values are handled
                df[col] = df[col].astype(int)
        
        # Step 6: Detect and handle outliers
        outlier_info = self.detect_outliers(df)
        df, outlier_treatment = self.handle_outliers(df, method=outlier_method, outlier_info=outlier_info)
        
        # Step 7: Create derived features
        df, derived_features = self.create_derived_features(df)
        
        # Step 8: Encode categorical variables
        df_final, encoding_info = self.encode_categorical_variables(df, encoding_method)
        
        # Store processed data
        self.processed_data = df_final
        
        # Create comprehensive report
        pipeline_report = {
            'original_shape': self.original_data.shape,
            'final_shape': df_final.shape,
            'type_changes': type_changes,
            'missing_value_treatment': missing_report,
            'outlier_info': outlier_info,
            'outlier_treatment': outlier_treatment,
            'derived_features': derived_features,
            'encoding_info': encoding_info,
            'processing_log': self.preprocessing_log
        }
        
        self.log_step("Full pipeline completed", f"Final shape: {df_final.shape}")
        
        return df_final, pipeline_report
    
    def prepare_for_modeling(self, test_size=0.2, random_state=42, scale_features=True):
        """Prepare data for machine learning models"""
        if self.processed_data is None:
            return None
        
        df = self.processed_data.copy()
        
        # Separate features and target
        if 'heart_disease' in df.columns:
            target_col = 'heart_disease'
        elif 'num' in df.columns:
            target_col = 'num'
        else:
            raise ValueError("No target variable found")
        
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        
        # Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Feature scaling
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            
            self.log_step("Features scaled", "StandardScaler applied")
        
        modeling_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'target_name': target_col
        }
        
        self.log_step("Data prepared for modeling", f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        return modeling_data
